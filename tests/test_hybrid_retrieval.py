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


def test_未配置语义路时走bigram词法路且守卫契约不变(monkeypatch):
    """没配 EMBED_API_KEY → 走 **bigram** 词法路，守卫契约不变。

    回归护栏：这里曾经断言 `retrieve("量子计算基础") == []`，那是空白分词时代的产物
    —— 它之所以成立，只是因为中文整句被切成一个 token、什么都匹配不上，**并不是
    拒答逻辑在起作用**。换成 bigram 后该查询会命中「计算机网络」，而标注集本身就把
    「量子计算基础」列为**争议样本**（产品语义可争议，不并入 C 组主口径）。
    所以改用真正语料外、无争议的例子。
    """
    monkeypatch.delenv("EMBED_BACKEND", raising=False)
    monkeypatch.setattr(emb, "get_embedding_provider", lambda: None)
    # ① 语料确实没有的主题 → 必须拒答
    assert rag.retrieve("摩托车发动机化油器怎么清洗") == []
    # ② 语料内主题 → 必须命中
    docs = rag.retrieve("LangGraph 多智能体编排入门")
    assert docs and docs[0].metadata["title"] == "LangGraph 多智能体编排入门"
    # ③ 中文同义改写也要能召回 —— 这正是空白分词做不到、bigram 才做到的
    assert rag.retrieve("零基础学门语言，先搞懂变量和循环"), "bigram 应能召回同义改写的中文问法"


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

        def embed(self, texts, is_query=False):
            raise RuntimeError("网络不可达")

    monkeypatch.setattr(emb, "get_embedding_provider", lambda: _Boom())
    # 降级后仍走 bigram 词法路：语料内仍能命中，语料外仍拒答
    assert rag.retrieve("LangGraph 多智能体编排入门") != []
    assert rag.retrieve("摩托车发动机化油器怎么清洗") == []


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

        def embed(self, texts, is_query=False):
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


def test_查询与文档的缓存键互相隔离(monkeypatch):
    """回归护栏：BGE 这类模型对 query 加检索指令、对文档不加，同一段文本在两种
    角色下的向量本来就不同。若查询与文档共用缓存键，先被当作文档嵌入过的文本
    再当查询用时会直接命中旧向量 —— 指令静默失效，而且指标上看不出来。"""
    calls: list[tuple[list[str], bool]] = []

    class _Recording:
        name = "recording"

        def embed(self, texts, is_query=False):
            calls.append((list(texts), is_query))
            return [[float(len(t))] * 8 for t in texts]

    monkeypatch.setattr(emb, "get_embedding_provider", lambda: _Recording())
    same = "FastAPI 怎么写接口"
    rag.retrieve(same, k=4)                      # 该文本先进语料侧（若有重合）与查询侧
    rag.retrieve(same, k=4)
    q_calls = [c for c in calls if c[1]]
    assert q_calls, "查询侧应以 is_query=True 调用（BGE 需要检索指令）"
    # 第二次检索同一查询不应再发查询侧请求
    assert sum(len(t) for t, iq in calls if iq) == 1, "查询向量缓存未生效"


def test_语义路失败后本进程不再反复重试(monkeypatch):
    """真实场景：账户欠费时每次检索都打一发注定失败的 HTTP 请求，既慢又刷日志。
    第一次失败就该记住并降级。"""
    attempts: list[int] = []

    class _Arrearage:
        name = "arrearage"

        def embed(self, texts, is_query=False):
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


# ───────────────────────── 重排序路径 ─────────────────────────


def _fake_embedding():
    class _E:
        name = "fake-embed"

        def embed(self, texts, is_query=False):
            # 只让含「LangGraph」的文档拿到正余弦，其余为 0 —— 制造「两路不一致」的局面
            return [[1.0] * 8 if "LangGraph" in t else [0.0] * 8 for t in texts]

    return _E()


def test_重排序路径用并集候选池(monkeypatch):
    """回归护栏：配了重排序器时，候选池必须是**并集**而不是交集。

    书里 L396 说融合产出「统一的候选池」；L387/L422 给的理由是两路盲区互补
    （稠密懂语义但漏关键词、稀疏精确匹配但读不懂同义词）。交集会丢掉
    「只有一路能召回」的文档 —— 那正是融合的价值所在。

    本用例构造两路各自只召回不同文档的局面，断言精排看到了**两路之和**。
    """
    from app import reranker as rk

    seen: dict = {}

    class _R:
        name = "fake-rerank"

        def score(self, query, docs):
            seen["n"] = len(docs)
            seen["docs"] = list(docs)
            return [0.9] * len(docs)          # 全部放行，只看候选池大小

    monkeypatch.setattr(emb, "get_embedding_provider", _fake_embedding)
    monkeypatch.setattr(rk, "get_reranker", lambda: _R())
    monkeypatch.delenv("ABLATE_THRESHOLD", raising=False)
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.3")

    rag.retrieve("LangGraph 多智能体编排入门", k=4)
    assert "n" in seen, "重排序器没被调用 —— 路径没走到"
    lex_only = [d for d in seen["docs"] if "LangGraph" not in d]
    # 并集应同时包含「词法独有」与「语义独有」两侧的文档
    assert len(seen["docs"]) >= 2, f"候选池过小：{len(seen['docs'])}"
    assert lex_only, "候选池里没有词法独有的文档 —— 可能退回了交集"


def test_重排序路径按绝对分拒答(monkeypatch):
    """拒答改由**绝对相关性分**决定（跨编码器同模型同权重，分数跨查询可比），
    不再依赖两路一致性 —— 所以「两路不一致但精排分高」的文档必须放行。"""
    from app import reranker as rk

    class _R:
        name = "fake-rerank"

        def score(self, query, docs):
            return [0.01] * len(docs)          # 全部低于阈值 → 应拒答

    monkeypatch.setattr(emb, "get_embedding_provider", _fake_embedding)
    monkeypatch.setattr(rk, "get_reranker", lambda: _R())
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.3")

    assert rag.retrieve("LangGraph 多智能体编排入门", k=4) == [], "全低于阈值应当拒答"


def test_重排序失败时退回RRF顺序而不是整体挂掉(monkeypatch):
    """重排序模型异常（缺文件/OOM）不能让检索整体失败 —— 退回 RRF 顺序。"""
    from app import reranker as rk

    class _Boom:
        name = "boom"

        def score(self, query, docs):
            raise RuntimeError("模型文件损坏")

    monkeypatch.setattr(emb, "get_embedding_provider", _fake_embedding)
    monkeypatch.setattr(rk, "get_reranker", lambda: _Boom())
    got = rag.retrieve("LangGraph 多智能体编排入门", k=4)
    assert got, "重排序失败后应退回 RRF 顺序，而不是返回空"


def test_没配重排序器时保持历史交集行为(monkeypatch):
    """默认（未配 RERANK_BACKEND）必须与历史行为一致 —— 「clone 即可跑」不能依赖 544 MB 模型。"""
    from app import reranker as rk

    monkeypatch.setattr(emb, "get_embedding_provider", _fake_embedding)
    monkeypatch.setattr(rk, "get_reranker", lambda: None)
    # 语义路对非 LangGraph 文档给 0 余弦，词法路给正分 → 交集可能为空
    got = rag.retrieve("Docker 镜像怎么打包", k=4)
    assert isinstance(got, list), "默认路径必须返回列表（交集判据可能为空）"
