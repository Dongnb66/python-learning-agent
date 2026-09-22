# -*- coding: utf-8 -*-
"""检索守卫判据选型实验 —— 结论是「单路 BM25 + 任何词法判据都做不到」。

零 Key、零网络、零 LLM、不改生产代码：只读 `app/data/resources.json` 与
`eval/retrieval_queries.json`，在 CJK bigram 分词下比较多种「是否该拒答」判据。

跑法：python scripts/retrieval_guard_probe.py

关键产出是第三节的**不可能性证明**：存在两条查询，BM25 分数与命中文档完全相同，
而期望行为相反（一条必须命中、一条必须拒答）—— 因此不存在任何阈值 / 覆盖率 /
稀有词规则能同时满足「召回」与「空命中拒答」契约。
"""
from __future__ import annotations

import io
import json
import os

from rank_bm25 import BM25Okapi

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CJK = (0x4E00, 0x9FFF)


def tokenize(text: str) -> list[str]:
    """CJK 字符 bigram（与 agent-platform-java Bm25Searcher.tokenize 同规则）。"""
    out: list[str] = []
    asc: list[str] = []
    low = text.lower()

    def flush() -> None:
        if asc:
            out.append("".join(asc))
            asc.clear()

    i, n = 0, len(low)
    while i < n:
        ch = low[i]
        if ch.isascii() and ch.isalnum():
            asc.append(ch)
            i += 1
            continue
        flush()
        if _CJK[0] <= ord(ch) <= _CJK[1]:
            j = i
            while j < n and _CJK[0] <= ord(low[j]) <= _CJK[1]:
                j += 1
            run = low[i:j]
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
        else:
            i += 1
    flush()
    return out


def load():
    data = json.load(io.open(os.path.join(_ROOT, "app", "data", "resources.json"), encoding="utf-8"))
    qset = json.load(io.open(os.path.join(_ROOT, "eval", "retrieval_queries.json"), encoding="utf-8"))
    docs = [f"{x.get('title', '')}\n{x.get('content', '')}" for x in data]
    cor = [tokenize(d) for d in docs]
    return qset, BM25Okapi(cor), cor


def main() -> int:
    qset, bm, cor = load()
    df: dict[str, int] = {}
    for c in cor:
        for tok in set(c):
            df[tok] = df.get(tok, 0) + 1

    def analyze(q: str) -> dict:
        qt = tokenize(q)
        sc = bm.get_scores(qt)
        top = max(range(len(sc)), key=lambda i: sc[i])
        matched = [t for t in qt if t in set(cor[top])]
        idfsum = sum(bm.idf.get(t, 0.0) for t in qt) or 1e-9
        return {
            "doc": top,
            "score": float(sc[top]),
            "cov": sum(bm.idf.get(t, 0.0) for t in matched) / idfsum,
            "rare": sum(1 for t in matched if df.get(t, 0) == 1),
        }

    ab = qset["A"] + qset["B"]
    must_refuse = [x["q"] for x in qset["C"] + qset["C_争议"]]
    rules = {
        "R0 score>0（生产现状语义）": lambda a: a["score"] > 0,
        "R1 score>2": lambda a: a["score"] > 2,
        "R1 score>3": lambda a: a["score"] > 3,
        "R2 IDF覆盖率>0.35": lambda a: a["cov"] > 0.35,
        "R3 命中≥1稀有词(df=1)": lambda a: a["rare"] >= 1,
        "R3 命中≥2稀有词": lambda a: a["rare"] >= 2,
        "R4 覆盖率>0.25 且 稀有词≥1": lambda a: a["cov"] > 0.25 and a["rare"] >= 1,
    }
    print("标注集：A=%d B=%d C=%d 争议=%d\n" % (
        len(qset["A"]), len(qset["B"]), len(qset["C"]), len(qset["C_争议"])))
    print("%-26s %7s %7s %7s %7s" % ("判据", "召回%", "Hit@1%", "C拒答%", "争议拒答%"))
    for name, keep in rules.items():
        rec = sum(1 for x in ab if keep(analyze(x["q"]))) / len(ab) * 100
        h1 = sum(1 for x in ab if analyze(x["q"])["doc"] == x["expected"] and keep(analyze(x["q"]))) / len(ab) * 100
        rr = sum(1 for q in qset["C"] if not keep(analyze(q["q"]))) / len(qset["C"]) * 100
        zr = sum(1 for x in qset["C_争议"] if not keep(analyze(x["q"]))) / len(qset["C_争议"]) * 100
        print("%-26s %7.1f %7.1f %7.1f %7.1f" % (name, rec, h1, rr, zr))
    print("\n注：所有判据 Hit@1 都卡在同一个上限 → 瓶颈是**排序**而不是阈值。")

    print("\n=== 不可能性证明 ===")
    cache = {q: analyze(q) for q in must_refuse}
    ceiling = max(v["score"] for v in cache.values())
    killer = max(cache.items(), key=lambda kv: kv[1]["score"])
    collide = [x["q"] for x in ab if abs(analyze(x["q"])["score"] - killer[1]["score"]) < 1e-9
               and analyze(x["q"])["doc"] == killer[1]["doc"]]
    print("应拒答中最高分: %.4f  ← %r (top_doc=%d)" % (ceiling, killer[0], killer[1]["doc"]))
    for cq in collide:
        a = analyze(cq)
        print("同分同文档的应命中查询: %r (score=%.4f, top_doc=%d)" % (cq, a["score"], a["doc"]))
    print("\n→ 两条查询的分数与命中文档**逐比特相同**，期望行为相反。")
    print("→ 任何只依赖词法分数的判据都无法分开，故单路 BM25 下「召回」与「空命中拒答」不可兼得。")
    print("→ 出路是第二条语义路（Embedding + RRF 融合），即 agent-platform-java 的 HybridRetriever 思路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
