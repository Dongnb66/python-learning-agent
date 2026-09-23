# -*- coding: utf-8 -*-
"""混合检索层单测 —— 零 Key、零网络、确定性。

⚠️ 这里用语义路的 `HashingEmbeddings`（字符 bigram 哈希），它**不具备语义能力**。
所以本文件只断言结构与降级行为（谁被放行、异常时退回哪条路、排序是否幂等），
**不**断言检索质量 —— 检索质量的度量在 `scripts/bm25_offline_bench.py` 与
`scripts/retrieval_guard_probe.py`，且真实语义路的数字需要配 `EMBED_API_KEY` 才能测。
"""
from __future__ import annotations

import pytest

from app import embeddings as emb
from app import rag


@pytest.fixture(autouse=True)
def _reset_caches():
    """每个用例前后清缓存，避免 provider 单例与语料向量跨用例串味。"""
    def _clear():
        rag.reset_retriever_cache()
        emb.reset_embedding_provider_cache()
    _clear()
    yield
    _clear()


def test_未配置语义路时行为与历史版本一致(monkeypatch):
    """没配 EMBED_API_KEY → 走空白分词 BM25，守卫契约不变。"""
    monkeypatch.delenv("EMBED_BACKEND", raising=False)
    monkeypatch.setattr(emb, "get_embedding_provider", lambda: None)
    assert rag.retrieve("量子计算基础") == []
    docs = rag.retrieve("LangGraph 多智能体编排入门")
    assert docs and docs[0].metadata["title"] == "LangGraph 多智能体编排入门"


def test_词法路零命中时混合路也必须拒答():
    """一条路都没命中 → 一致性判据必然为空 → 返回空列表，而不是硬塞 top-k。"""
    assert rag._lexical_scores(rag.get_retriever(k=4), "🐉🦄✨") == [0.0] * len(rag._all_docs())
    assert rag.retrieve("🐉🦄✨") == []


def test_启用语义路后放行结果仍是语料子集且幂等(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    assert isinstance(emb.get_embedding_provider(), emb.HashingEmbeddings)
    docs_a = rag.retrieve("FastAPI 怎么写接口", k=3)
    docs_b = rag.retrieve("FastAPI 怎么写接口", k=3)
    urls = {d.metadata["url"] for d in rag._all_docs()}
    assert all(d.metadata["url"] in urls for d in docs_a), "放行项必须来自语料，不能是生成的"
    assert [d.metadata["url"] for d in docs_a] == [d.metadata["url"] for d in docs_b], "排序须幂等"


def test_混合路必须真的放行语料内查询(monkeypatch):
    """回归护栏：曾因把 embedding 的「向量列表」当单个向量传给 cosine，导致长度不等、
    余弦恒为 0、一致性判据把 20/20 全拒 —— 而上面那条「子集 + 幂等」测试对空列表同样成立，
    抓不到。故必须显式断言非空。"""
    monkeypatch.setenv("EMBED_BACKEND", "hashing")
    for q, title in (("LangGraph 多智能体编排入门", "LangGraph 多智能体编排入门"),
                     ("RAG 检索增强生成的原理", "RAG 检索增强生成原理与实践")):
        got = rag.retrieve(q, k=4)
        assert got, f"语料内查询被混合路误拒：{q}"
        assert got[0].metadata["title"] == title


def test_embedding异常时降级为词法路(monkeypatch):
    class _Boom:
        name = "boom"

        def embed(self, texts):
            raise RuntimeError("网络不可达")

    monkeypatch.setattr(emb, "get_embedding_provider", lambda: _Boom())
    # 降级后等价于历史行为：语料内仍能命中，语料外仍拒答
    assert rag.retrieve("LangGraph 多智能体编排入门") != []
    assert rag.retrieve("量子计算基础") == []


def test_bigram分词对中文产出多token():
    toks = rag.tokenize("怎么把项目打包成镜像")
    assert len(toks) > 5, "bigram 应把中文切成多个二元组，而不是整句一个 token"
    assert "怎么" in toks and "镜像" in toks
    assert rag.tokenize("FastAPI 接口") [0] == "fastapi", "ASCII 段应转小写按词切"


def test_查询向量缓存不重复打网络(monkeypatch):
    """同一查询被 Hit@1/3/5 + MRR + 拒答判定反复调用，评测一轮就是数百次请求。
    缓存必须保证每段文本只发一次 —— 这条断言直接数发出去的文本。"""
    sent: list[list[str]] = []

    class _Counting:
        name = "counting"

        def embed(self, texts):
            batch = list(texts)
            sent.append(batch)
            return [[0.0] * 8 for _ in batch]

    monkeypatch.setattr(emb, "get_embedding_provider", lambda: _Counting())
    for _ in range(3):
        rag.retrieve("FastAPI 怎么写接口", k=4)
        rag.retrieve("LangGraph 怎么做多智能体编排", k=4)
    flat = [t for batch in sent for t in batch]
    n_docs = len(rag._all_docs())
    assert len(flat) == n_docs + 2, f"应只发 {n_docs} 篇语料 + 2 条查询，实发 {len(flat)}"
    assert len(set(flat)) == len(flat), "存在重复发送的文本，缓存未生效"


def test_语义路失败后本进程不再反复重试(monkeypatch):
    """真实场景：账户欠费时每次检索都打一发注定失败的 HTTP 请求，既慢又刷日志。
    第一次失败就该记住并降级。"""
    attempts: list[int] = []

    class _Arrearage:
        name = "arrearage"

        def embed(self, texts):
            attempts.append(len(texts))
            raise RuntimeError("400 Arrearage")

    monkeypatch.setattr(emb, "get_embedding_provider", lambda: _Arrearage())
    for _ in range(20):
        assert rag.retrieve("LangGraph 多智能体编排入门") != []   # 降级后仍走词法路，管线不断
    assert len(attempts) == 1, f"失败后仍重试了 {len(attempts)} 次"


def test_显式关闭语义路优先于已配置的Key(monkeypatch):
    """回归护栏：曾有逻辑是 `backend in ("","none") and not embed_configured()`，
    于是本机 .env 配了 key 时 `EMBED_BACKEND=none` 反而不生效 —— 测试模式会被
    秘密文件悄悄改掉。现在 none 必须无条件关闭语义路。"""
    monkeypatch.setenv("EMBED_BACKEND", "none")
    monkeypatch.setattr(emb, "embed_configured", lambda: True)
    assert emb.get_embedding_provider() is None


def test_消融开关仍然绕过一致性判据(monkeypatch):
    """ABLATE_THRESHOLD=1 时退回 BM25Retriever 原生行为（含 0 分项）—— 对照实验依赖它。"""
    monkeypatch.setenv("ABLATE_THRESHOLD", "1")
    got = rag.retrieve("🐉🦄✨", k=4)
    assert len(got) == 4, "关掉守卫后应无条件返回 top-k，哪怕全是 0 分项"
