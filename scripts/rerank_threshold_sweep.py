# -*- coding: utf-8 -*-
"""重排序绝对分阈值扫描 —— 找出「并集候选池 + 精排」在本地语料上的最优放行阈值。

为什么必须扫而不能拍
--------------------
第一次跑重排序路径用的是默认阈值 `RERANK_MIN_SCORE=0.30`，结果合计 Hit@1 只有
47.2%、误拒 19/36（52.8%）—— 阈值把一半该放行的查询挡掉了。

BM25 的 `min_score=2.0` 当初也是扫出来的（见 `retrieval_guard_probe.py` 的判据选型表）。
本脚本对重排序分做同样的事：扫描阈值，同时报 **命中率**（A+B）与**C 组正确拒答率**，
让「拿召回换拒答」的取舍显式可见 —— 而不是拍一个数然后宣称它有效。

用法：
    EMBED_BACKEND=local-onnx EMBED_LOCAL_MODEL_DIR=<embed>
    RERANK_BACKEND=local-onnx RERANK_LOCAL_MODEL_DIR=<rerank>
    python scripts/rerank_threshold_sweep.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import rag  # noqa: E402
from app.embeddings import cosine, get_embedding_provider  # noqa: E402
from app.reranker import get_reranker  # noqa: E402

K_PROD = 4
THRESHOLDS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def main() -> int:
    corpus = rag._load_documents()
    url2idx = {d.metadata.get("url", ""): i for i, d in enumerate(corpus)}
    qs = json.load(open(os.path.join(_ROOT, "eval", "retrieval_queries.json"), encoding="utf-8"))
    labeled = [(x["q"], x["expected"]) for x in qs["A"]] + [(x["q"], x["expected"]) for x in qs["B"]]
    c_list = [x["q"] for x in qs["C"]]

    provider = get_embedding_provider()
    reranker = get_reranker()
    if provider is None or reranker is None:
        print("需要同时配好 EMBED_* 与 RERANK_* 才能扫描。", file=sys.stderr)
        return 2
    print(f"embed = {provider.name}\nrerank = {reranker.name}\nn = {len(labeled)} 有答案 / {len(c_list)} 语料外\n")

    doc_vecs = provider.embed([d.page_content for d in corpus])
    bigram = rag._get_bigram_retriever(k=K_PROD)

    def candidates(q):
        """复刻生产路径的候选池构造：两路 top-k 并集 → RRF 排序。"""
        lex = rag._lexical_scores(bigram, q)
        qv = provider.embed([q], is_query=True)[0]
        sem = [cosine(qv, dv) for dv in doc_vecs]

        def topk(sc):
            ix = [i for i, _ in sorted(enumerate(sc), key=lambda p: -p[1])[:K_PROD]]
            return [i for i in ix if sc[i] > 0]

        lt, st = topk(lex), topk(sem)
        pool = list(dict.fromkeys(lt + st))

        def rrf(i):
            s = 0.0
            if i in lt:
                s += 1.0 / (rag._RRF_K + lt.index(i) + 1)
            if i in st:
                s += 1.0 / (rag._RRF_K + st.index(i) + 1)
            return s

        return sorted(pool, key=rrf, reverse=True)

    # 一次算好每个查询的 (候选下标, 精排分)，之后只按阈值过滤 —— 否则要跑 10 遍模型
    cache: dict[str, list[tuple[int, float]]] = {}
    for q, _ in labeled:
        c = candidates(q)
        cache[q] = list(zip(c, reranker.score(q, [corpus[i].page_content for i in c]))) if c else []
    for q in c_list:
        c = candidates(q)
        cache[q] = list(zip(c, reranker.score(q, [corpus[i].page_content for i in c]))) if c else []

    print(f"{'阈值':>6}{'A+B命中':>9}{'Hit@1':>8}{'Hit@5':>8}{'MRR':>8}{'误拒':>7}{'C组拒答':>10}")
    best = None
    for t in THRESHOLDS:
        h1 = h5 = 0
        mrr = 0.0
        false_rej = 0
        all_scores: list[float] = []
        for q, exp in labeled:
            kept = sorted([(i, s) for i, s in cache[q] if s >= t], key=lambda p: -p[1])
            order = [i for i, _ in kept]
            if order[:1] == [exp]:
                h1 += 1
            if exp in order[:5]:
                h5 += 1
            if exp in order:
                mrr += 1.0 / (order.index(exp) + 1)
            else:
                false_rej += 1
        c_ok = sum(1 for q in c_list if not [1 for _i, s in cache[q] if s >= t])
        n = len(labeled)
        row = (t, 100 * h1 / n, 100 * h5 / n, mrr / n, false_rej, 100 * c_ok / len(c_list))
        print(f"{t:>6.2f}{h1:>9}{row[1]:>7.1f}%{row[2]:>7.1f}%{row[3]:>8.3f}{false_rej:>7}{row[5]:>9.1f}%")
        # 「不漏编造」优先：先满足 C 组拒答 >= 95%，再比 Hit@1
        if row[5] >= 95.0 and (best is None or row[1] > best[1]):
            best = row

    print()
    if best:
        print(f"建议阈值 ≈ {best[0]:.2f}（C 组拒答 {best[5]:.1f}% 前提下 Hit@1 最高 {best[1]:.1f}%）")
    else:
        print("没有任何阈值能在保住 C 组拒答 95% 的同时取得可用命中率 —— "
              "说明本地语料（12 篇 / 849 字）上，重排序的绝对分区分度不足以做拒答判据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
