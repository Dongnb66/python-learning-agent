# -*- coding: utf-8 -*-
"""融合机制消融探针：分离「两路一致性交集」与「RRF 排序」各自的贡献。

背景
----
用真语义模型跑 `bm25_offline_bench.py` 时，合计 Hit@1 从纯词法的 77.8% 掉到 52.8%，
但 Hit@5 升到 91.7%、C 组拒答升到 100%。怀疑根因是 `app/rag.py` 里这一行：

    agree = [i for i in lex_top if i in sem_top]     # 两路 top-k 交集

真语义路的 top-k 与词法路差异大 → 交集收缩 → 词法排第一的文档常被滤掉，
于是 Top-1 掉、但召回被语义路补上来了。

四种策略（L/A 直接调生产函数，保证保真）：
  L  纯词法（生产 `_legacy_retrieve`，带 min_score 阈值）
  S  纯语义 top-k（自实现）
  A  交集 + RRF（**生产 `_hybrid_retrieve`**）
  U  并集 + RRF（自实现，去掉一致性过滤）

用法：
    python scripts/fusion_probe.py
    EMBED_BACKEND=local-onnx EMBED_LOCAL_MODEL_DIR=<目录> python scripts/fusion_probe.py
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

K_PROD = 4
MIN_SCORE_PROD = 2.0


def rank_of(scores, k, positive_only=True):
    idx = [i for i, _ in sorted(enumerate(scores), key=lambda p: -p[1])[:k]]
    return [i for i in idx if scores[i] > 0] if positive_only else idx


def main() -> int:
    corpus = rag._load_documents()
    url2idx = {d.metadata.get("url", ""): i for i, d in enumerate(corpus)}
    data = json.load(open(os.path.join(_ROOT, "eval", "retrieval_queries.json"), encoding="utf-8"))
    provider = get_embedding_provider()
    print(f"语料 {len(corpus)} 篇   provider = {provider.name if provider else 'None（纯词法）'}")
    print("分组: " + "  ".join(f"{k}={len(v)}" for k, v in data.items() if isinstance(v, list)))
    print()

    bigram = rag._get_bigram_retriever(k=K_PROD)
    doc_vecs = provider.embed([d.page_content for d in corpus]) if provider else None

    def as_idx(docs):
        return [url2idx.get(d.metadata.get("url", ""), -1) for d in docs]

    def order_L(q):
        return [i for i in as_idx(rag._legacy_retrieve(q, K_PROD, MIN_SCORE_PROD)) if i >= 0]

    def order_S(q):
        if doc_vecs is None:
            return []
        qv = provider.embed([q], is_query=True)[0]
        return rank_of([cosine(qv, dv) for dv in doc_vecs], K_PROD)

    def order_A(q):
        if doc_vecs is None:
            return order_L(q)
        r = rag._hybrid_retrieve(q, K_PROD)
        return [] if r is None else [i for i in as_idx(r) if i >= 0]

    def order_U(q):
        if doc_vecs is None:
            return order_L(q)
        lex = rag._lexical_scores(bigram, q)
        qv = provider.embed([q], is_query=True)[0]
        sem = [cosine(qv, dv) for dv in doc_vecs]
        lt, st = rank_of(lex, K_PROD), rank_of(sem, K_PROD)
        union = list(dict.fromkeys(lt + st))

        def rrf(i):
            s = 0.0
            if i in lt:
                s += 1.0 / (rag._RRF_K + lt.index(i) + 1)
            if i in st:
                s += 1.0 / (rag._RRF_K + st.index(i) + 1)
            return s

        return sorted(union, key=rrf, reverse=True)

    strategies = {"L 纯词法": order_L, "S 纯语义": order_S, "A 交集+RRF（生产）": order_A, "U 并集+RRF": order_U}
    stats = {k: {"n": 0, "h1": 0, "h3": 0, "h5": 0, "mrr": 0.0} for k in strategies}
    refuse = {k: [0, 0] for k in strategies}

    for gname in ("A", "B", "C", "C_争议"):
        for item in data.get(gname, []):
            q = item["q"]
            exp = item.get("expected")
            for name, fn in strategies.items():
                order = fn(q)
                if gname == "C":
                    refuse[name][1] += 1
                    if not order:                       # 语料外 → 生产口径「返回空 = 正确拒答」
                        refuse[name][0] += 1
                    continue
                if exp is None:
                    continue
                st = stats[name]
                st["n"] += 1
                if order[:1] == [exp]:
                    st["h1"] += 1
                if exp in order[:3]:
                    st["h3"] += 1
                if exp in order[:5]:
                    st["h5"] += 1
                if exp in order:
                    st["mrr"] += 1.0 / (order.index(exp) + 1)

    print(f"{'策略':<22}{'n':>4}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'MRR':>8}{'C组拒答':>10}")
    for name, st in stats.items():
        if not st["n"]:
            continue
        ok, tot = refuse[name]
        print(f"{name:<22}{st['n']:>4}{100*st['h1']/st['n']:>8.1f}%{100*st['h3']/st['n']:>8.1f}%"
              f"{100*st['h5']/st['n']:>8.1f}%{st['mrr']/st['n']:>8.3f}"
              f"{(100*ok/tot if tot else 0):>9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
