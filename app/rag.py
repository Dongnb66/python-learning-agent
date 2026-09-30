"""RAG 检索层（防幻觉）。

两条路，一个契约
----------------
- **词法路**：BM25（langchain_community + rank_bm25）。
- **语义路**：Embedding 余弦相似度（可选，需 `EMBED_API_KEY`）。

契约始终是同一条：**资料库里确实没有相关内容时返回空列表**，让上层能在代码层
拒答，而不是把无关材料塞给模型、反过来助推它编造。

为什么需要第二条路（离线基准实测，见 `scripts/retrieval_guard_probe.py`）：
单路 BM25 下「写 HTTP 接口的 Python 框架推荐」与「量子计算基础」会拿到**完全相同**
的分数并命中同一篇文档，而前者该命中、后者该拒答 —— 任何绝对阈值 / IDF 覆盖率 /
稀有词计数都分不开它们。因此混合模式改用**两路一致性**放行：只有词法与语义都认可
的资料才作为「事实依据」交给模型。

词法路的分词
------------
**两条路都统一用 CJK bigram 分词**（`tokenize()`）。

这里曾经是「未配 Key 就退回空白分词，与历史版本逐行为一致」。实测证明那是个缺陷：
中文没有空格，空白分词会把整句切成**一个** token，于是标注集 B 组（同义改写 24 条）
Hit@1 = 0%、**所有中文问法都被「空命中拒答」挡掉** —— 也就是「空命中拒答」退化成了
「中文一律拒答」，而 C 组 100% 的拒答率只是这个故障的另一面。

bigram 是纯本地算法，**不需要 Key、不需要网络**，没有任何理由把它挂在语义路后面。

实测（`scripts/bm25_offline_bench.py` 口径，`min_score=2.0`）：
合计 Hit@1 **22.2% → 77.8%**，C 组正确拒答 95.5%。

未配置 Embedding Key 时，`retrieve()` 走这条词法路；语义路是可选增强，不是运行前提。
"""
from __future__ import annotations

import json
import os

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from app import telemetry

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "resources.json")

_CJK_START, _CJK_END = 0x4E00, 0x9FFF
_RRF_K = 60            # RRF 平滑常数


def tokenize(text: str) -> list[str]:
    """CJK 字符 bigram 分词，与 agent-platform-java 的 Bm25Searcher.tokenize 同一套规则。

    只有混合模式用它。默认空白分词对中文无效：中文没有空格，整句被切成**一个**
    token，「怎么把项目打包成镜像跑起来」这类自然问法对语料永远不命中（实测同义
    改写组 Hit@1 = 0%），于是「空命中拒答」退化成「中文一律拒答」。
    """
    tokens: list[str] = []
    asc: list[str] = []
    low = text.lower()

    def flush() -> None:
        if asc:
            tokens.append("".join(asc))
            asc.clear()

    i, n = 0, len(low)
    while i < n:
        ch = low[i]
        if ch.isascii() and ch.isalnum():
            asc.append(ch)
            i += 1
            continue
        flush()
        if _CJK_START <= ord(ch) <= _CJK_END:
            j = i
            while j < n and _CJK_START <= ord(low[j]) <= _CJK_END:
                j += 1
            run = low[i:j]
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
        else:
            i += 1
    flush()
    return tokens


def _load_documents() -> list[Document]:
    with open(_CORPUS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        # 把元数据拼进正文，提升 BM25 命中率
        text = f"{item.get('title', '')}\n{item.get('content', '')}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "type": item.get("type", "article"),
                },
            )
        )
    return docs


_retriever: BM25Retriever | None = None
_bigram_retriever: BM25Retriever | None = None
_docs: list[Document] | None = None
_doc_vectors: list[list[float]] | None = None
_embed_cache: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 512
_embed_unavailable = False   # 语义路一旦确认不可用就不再重试，避免每次检索都打一次注定失败的请求


def get_retriever(k: int = 4) -> BM25Retriever:
    """历史入口：默认（空白）分词的 BM25，行为与引入混合检索前一致。"""
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever.from_documents(_load_documents(), k=k)
    return _retriever


def _all_docs() -> list[Document]:
    global _docs
    if _docs is None:
        _docs = _load_documents()
    return _docs


def _get_bigram_retriever(k: int = 4) -> BM25Retriever:
    global _bigram_retriever
    if _bigram_retriever is None:
        _bigram_retriever = BM25Retriever.from_documents(
            _all_docs(), k=k, preprocess_func=tokenize
        )
    return _bigram_retriever


def reset_retriever_cache() -> None:
    """清掉检索层缓存 —— 供测试与语料热更新使用。"""
    global _retriever, _bigram_retriever, _docs, _doc_vectors, _embed_cache, _embed_unavailable
    _retriever = _bigram_retriever = None
    _docs = _doc_vectors = None
    _embed_cache.clear()
    _embed_unavailable = False


def _guard_disabled() -> bool:
    """消融实验开关：`ABLATE_THRESHOLD=1` 时关掉相关性阈值过滤。

    仅供 `scripts/run_ablation.py` 做对照实验使用，生产环境不会设置该变量。
    """
    return os.getenv("ABLATE_THRESHOLD", "") == "1"


def _rerank_candidates() -> int:
    """精排前先截断到多少个候选。

    ⚠️ **在调用时才读环境变量，不在模块级捕获。**
    《深入理解 AI Agent》第 7 章 L785 明确警告过这个坑：
    「消融开关必须在启动路径的**极早期**注入 —— 在任何模块级常量捕获配置值之前」。
    模块级读取会让「先 import 再设环境变量」的消融实验静默失效。
    """
    try:
        return max(1, int(os.getenv("RERANK_CANDIDATES", "50")))
    except ValueError:
        return 50


def _rerank_min_score() -> float:
    """绝对相关性分阈值 —— 低于它就不作为事实依据（拒答）。

    为什么这里可以用绝对分：跨编码器是**同一个模型、同一套权重**给每对
    (query, doc) 打分，分数跨查询可比。而 BM25 分与余弦相似度都跨查询/跨模型
    不可比（本仓库实测：换成哈希向量后余弦全落在 0.18 以下 → 20/20 全误拒），
    所以那两条路只能靠名次，不能靠阈值。
    """
    try:
        return float(os.getenv("RERANK_MIN_SCORE", "0.30"))
    except ValueError:
        return 0.30


def _lexical_scores(retriever: BM25Retriever, query: str) -> list[float]:
    return list(retriever.vectorizer.get_scores(retriever.preprocess_func(query)))


def _legacy_retrieve(query: str, k: int, min_score: float) -> list[Document]:
    """纯词法路（未启用语义路时的唯一路径）。

    用 **bigram** 分词，不用空白分词：中文没有空格，空白分词会把整句切成一个 token，
    中文查询永远零命中 —— 那不是「正确拒答」，是「中文检索不工作」。

    BM25 分数为 0 表示「query 的词一个都没在语料里命中」——这类结果只是凑数。
    BM25Retriever 原生无论如何都返回 top-k（含 0 分项），直接当「事实依据」用
    等于给模型塞无关材料，反而助推编造，所以这里显式过滤：确实没有相关内容时
    返回空列表，上层据此**在代码层拒答**，而不是指望模型自觉。
    """
    r = _get_bigram_retriever(k=k)
    scores = _lexical_scores(r, query)
    ranked = sorted(zip(scores, r.docs), key=lambda pair: pair[0], reverse=True)
    telemetry.bump(telemetry.C_RETRIEVAL)
    if _guard_disabled():
        return [doc for _score, doc in ranked[:k]]
    hits = [doc for score, doc in ranked[:k] if score > min_score]
    # 第①道守卫的运行时证据：阈值过滤后为空的次数（= 该话题资料库确实没有）
    if not hits:
        telemetry.bump(telemetry.C_RETRIEVAL_EMPTY)
    return hits


def _embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]] | None:
    """按文本缓存 embedding，语义路不可用或调用失败时返回 None。

    缓存是必需的而非优化：一次检索评测会对同一批查询重复请求数百次
    （Hit@1/3/5 + MRR + 拒答判定各一次，再加延迟采样循环），不缓存就是把
    同一句话反复发给远程 API —— 既烧额度，测出来的延迟也全是网络抖动。

    `is_query=True` 走独立缓存键：BGE 这类模型对查询加检索指令、对文档不加，
    同一段文本在两种角色下的向量本来就不同，混用缓存会串味。
    """
    from app.embeddings import get_embedding_provider

    global _embed_unavailable
    provider = get_embedding_provider()
    if provider is None or _embed_unavailable:
        return None
    ck = (lambda t: "q:" + t) if is_query else (lambda t: "d:" + t)
    todo = [t for t in dict.fromkeys(texts) if ck(t) not in _embed_cache]
    if todo:
        try:
            fresh = provider.embed(todo, is_query=is_query)
        except Exception as exc:  # 网络 / 额度 / 模型异常 → 降级并记住，不再反复重试
            telemetry.bump(telemetry.C_RETRIEVAL_EMBED_FAIL)
            _embed_unavailable = True
            print(f"[RAG] Embedding 调用失败，本进程内退回纯词法检索：{exc}")
            return None
        if len(fresh) != len(todo):
            telemetry.bump(telemetry.C_RETRIEVAL_EMBED_FAIL)
            return None
        for text, vec in zip(todo, fresh):
            _embed_cache[ck(text)] = vec
        while len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.pop(next(iter(_embed_cache)))
        telemetry.bump(telemetry.C_EMBED_CALLS, len(todo))
    return [_embed_cache[ck(t)] for t in texts]


def _doc_vectors_or_none() -> list[list[float]] | None:
    global _doc_vectors
    if _doc_vectors is None:
        vectors = _embed_texts([d.page_content for d in _all_docs()])
        if vectors is None:
            return None
        _doc_vectors = vectors
    return _doc_vectors


def _hybrid_retrieve(query: str, k: int) -> list[Document] | None:
    """BM25(bigram) + Embedding 两路召回 → 融合 →（可选）重排序 → 放行判据。

    **两条放行策略，取决于是否配了重排序器：**

    * **有重排序器**（`RERANK_BACKEND=local-onnx`）—— 书里 L396-398 的标准流水线：
      两路结果取**并集**当候选池 → RRF 排序 → **跨编码器精排** →
      用**绝对相关性分**卡阈值。并集保住「只有一路能召回」的文档；
      跨编码器输出的是绝对分，解决「BM25 分与余弦都跨查询不可比」导致的拒答难题。
    * **无重排序器** —— 退回历史行为：**交集一致性**放行。
      它拒答可靠（实测 C 组 100%），但代价是丢掉单路独有命中（实测 n=36 少 2 条）。

    返回 None 表示语义路不可用（未配 Key 或调用失败），调用方退回纯词法路。
    """
    from app.embeddings import cosine
    from app.reranker import get_reranker

    doc_vecs = _doc_vectors_or_none()
    if doc_vecs is None:
        return None
    qvecs = _embed_texts([query], is_query=True)
    if not qvecs:
        return None
    qvec = qvecs[0]
    if not doc_vecs or len(qvec) != len(doc_vecs[0]):
        # 提供者换型（不同维度模型）后语料向量已失效 —— 降级而不是静默算出 0 分
        telemetry.bump(telemetry.C_RETRIEVAL_EMBED_FAIL)
        return None

    docs = _all_docs()
    bm25 = _get_bigram_retriever(k=k)
    lex = _lexical_scores(bm25, query)
    sem = [cosine(qvec, dv) for dv in doc_vecs]
    telemetry.bump(telemetry.C_RETRIEVAL)

    lex_top = [i for i, _ in sorted(enumerate(lex), key=lambda p: -p[1])[:k] if lex[i] > 0]
    # 语义路用 top-k **名次**而非绝对余弦当门槛。理由实测过：绝对分数跨模型不可比 ——
    # 把 provider 换成字符哈希向量后所有余弦都落在 0.18 以下，导致 20/20 全部误拒；
    # 名次天然尺度无关，也正是绕开「同分冲突」的那把钥匙。
    sem_top = [i for i, _ in sorted(enumerate(sem), key=lambda p: -p[1])[:k] if sem[i] > 0]

    reranker = get_reranker()

    if reranker is None:
        # ── 历史路径：交集一致性放行 ──
        agree = [i for i in lex_top if i in sem_top]
        if not agree:
            telemetry.bump(telemetry.C_RETRIEVAL_EMPTY)
            return []

        def rrf_agree(i: int) -> float:
            a = lex_top.index(i) + 1
            b = sem_top.index(i) + 1
            return 1.0 / (_RRF_K + a) + 1.0 / (_RRF_K + b)

        return [docs[i] for i in sorted(agree, key=rrf_agree, reverse=True)[:k]]

    # ── 重排序路径：并集候选池 → RRF → 跨编码器精排 → 绝对分阈值 ──
    # 候选池用并集：出现在任一路的文档都进来。书里 L387/L422 的理由是两路盲区互补
    # （稠密懂语义但漏关键词、稀疏精确匹配但读不懂同义词），交集恰好丢掉这部分。
    pool = list(dict.fromkeys(lex_top + sem_top))
    if not pool:
        telemetry.bump(telemetry.C_RETRIEVAL_EMPTY)
        return []

    def rrf_pool(i: int) -> float:
        """标准 RRF：只在文档出现的那一路累加（缺席不补大名次）。"""
        s = 0.0
        if i in lex_top:
            s += 1.0 / (_RRF_K + lex_top.index(i) + 1)
        if i in sem_top:
            s += 1.0 / (_RRF_K + sem_top.index(i) + 1)
        return s

    ordered = sorted(pool, key=rrf_pool, reverse=True)
    # 精排只需要前 N 个候选；池子通常远小于 N，但语料变大后这一步是必要的截断
    head = ordered[: _rerank_candidates()]
    try:
        scores = reranker.score(query, [docs[i].page_content for i in head])
    except Exception as exc:  # 重排序失败不能让检索整体挂掉 —— 退回 RRF 顺序
        telemetry.bump(telemetry.C_RETRIEVAL_EMBED_FAIL)
        print(f"[RAG] 重排序失败，本次退回 RRF 顺序：{exc}")
        return [docs[i] for i in ordered[:k]]

    if len(scores) != len(head):
        telemetry.bump(telemetry.C_RETRIEVAL_EMBED_FAIL)
        return [docs[i] for i in ordered[:k]]

    # 绝对分阈值拒答：跨编码器同一个模型、同一套权重，分数跨查询可比 ——
    # 这正是它相对 BM25 分/余弦的优势（后两者跨查询不可比，只能靠名次或交集）。
    floor = _rerank_min_score()
    kept = [(i, s) for i, s in zip(head, scores) if s >= floor]
    if not kept:
        telemetry.bump(telemetry.C_RETRIEVAL_EMPTY)
        return []
    return [docs[i] for i, _ in sorted(kept, key=lambda p: -p[1])[:k]]


def retrieve(query: str, k: int = 4, min_score: float = 2.0) -> list[Document]:
    """返回与 query 最相关的资料片段（按相关性降序）；语料确实没有则返回空列表。

    `min_score` 仅在未启用语义路时生效。默认 **2.0** 是实测选出来的，不是拍的：
    在 bigram 分词下把「语料外查询被误放行」从 2/22 压到 1/22（C 组正确拒答
    90.9% → 95.5%），而合计 Hit@1 保持 77.8% 不变 —— **零召回代价换来拒答提升**。
    （对比数据见 `scripts/retrieval_guard_probe.py` 的判据选型表。）

    启用混合检索后放行判据改为「两路 top-k 一致性」而非绝对分数门槛，因为 BM25 分数
    与余弦相似度都跨查询/跨模型不可比 —— 前者由 `scripts/retrieval_guard_probe.py`
    的同分冲突实测证明，后者由本文件那次 20/20 误拒实测证明。
    """
    if not _guard_disabled():
        hybrid = _hybrid_retrieve(query, k)
        if hybrid is not None:
            return hybrid
    return _legacy_retrieve(query, k, min_score)
