# -*- coding: utf-8 -*-
"""检索层离线基准（零 API Key、零网络、零 LLM）。

为什么能离线：BM25 是确定性算法，检索层本身不调模型，所以命中率与延迟都是
**真实测量值**，不是 mock 出来的 —— 与 run_eval.py 的端到端评测互补。

查询集来自 `eval/retrieval_queries.json`（唯一真值来源，本脚本不再内联副本）：
A 组=语料内直述，B 组=语料内但词面不匹配（同义改写），C 组=语料外应拒答，
C_争议=产品语义可争议或主要混淆源，单列不并入 C 的主口径。

⚠️ 四条必须一起读的局限（否则数字会被误读）：
  1. 语料仅 12 篇 / 849 字（正文净字数）。单相关文档下随机基线 Hit@5 = 5/12 ≈ 41.7%，
     只报绝对命中率没有意义，故本脚本同时报随机基线。
     注：报告字段「正文总字数」为历史命名，实为 page_content 合计 = 标题 + 换行 + 正文 = 1065。
  2. 标注由 AI agent（Qoder）生成，非人工、未独立复核（弱监督）。
  3. 测的是**检索层排序行为**，不是真实用户满意度。
  4. 若用 `EMBED_BACKEND=hashing` 跑，语义路用的是字符哈希向量，**不具备语义
     能力**，那一列只能证明链路结构正确，不得对外称检索质量。

用法：python scripts/bm25_offline_bench.py
输出：终端表格 + eval/bm25_bench_<时间戳>.json
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import rag  # noqa: E402  —— 直接用生产代码路径，不重写 BM25

# 生产参数：app/agents/resource_agent.py 用 k=4、min_score 默认 0.0
K_PROD = 4
MIN_SCORE_PROD = 0.0
_QUERIES_FILE = os.path.join(_ROOT, "eval", "retrieval_queries.json")

_URL2IDX: dict[str, int] = {}


def doc_index(doc) -> int:
    """把返回的 Document 映射回语料下标（按 url 对齐）。"""
    return _URL2IDX.get(doc.metadata.get("url", ""), -1)


def hit_at_k(query: str, expected: int, k: int) -> bool:
    return any(doc_index(d) == expected
               for d in rag.retrieve(query, k=k, min_score=MIN_SCORE_PROD))


def mrr(query: str, expected: int) -> float:
    for rank, d in enumerate(rag.retrieve(query, k=10, min_score=MIN_SCORE_PROD), 1):
        if doc_index(d) == expected:
            return 1.0 / rank
    return 0.0


def refused(query: str) -> bool:
    return not rag.retrieve(query, k=K_PROD, min_score=MIN_SCORE_PROD)


def main() -> int:
    global _URL2IDX
    corpus = rag._load_documents()
    _URL2IDX = {d.metadata.get("url", ""): i for i, d in enumerate(corpus)}
    n = len(corpus)

    qs = json.load(io.open(_QUERIES_FILE, encoding="utf-8"))
    labeled = [("A", x["q"], x["expected"]) for x in qs["A"]] + \
              [("B", x["q"], x["expected"]) for x in qs["B"]]
    c_list = [x["q"] for x in qs["C"]]
    z_list = [x["q"] for x in qs["C_争议"]]
    allq = len(labeled)

    # ---- 分词诊断：报告本次实际生效的那条路 ----
    from app.embeddings import get_embedding_provider
    sample = "Python 变量和循环怎么入门"
    hybrid = get_embedding_provider() is not None
    active = rag._get_bigram_retriever(k=K_PROD) if hybrid else rag.get_retriever(k=K_PROD)
    diag = {"query": sample, "tokens": list(active.preprocess_func(sample)),
            "生效路径": "混合（bigram BM25 + Embedding + RRF + 两路一致性）" if hybrid else "纯词法（空白分词 BM25）",
            "tokenizer": getattr(active.preprocess_func, "__name__", "callable")}

    t0 = time.perf_counter()
    rag.retrieve("预热 warmup", k=K_PROD)
    build_ms = (time.perf_counter() - t0) * 1000

    # ---- 命中率 / MRR / 误拒 ----
    pct = lambda a, b: round(100.0 * a / b, 1) if b else 0.0  # noqa: E731
    groups = {}
    for grp in ("A", "B"):
        items = [(q, e) for g, q, e in labeled if g == grp]
        m = len(items)
        groups[grp] = {
            "n": m,
            "Hit@1%": pct(sum(hit_at_k(q, e, 1) for q, e in items), m),
            "Hit@3%": pct(sum(hit_at_k(q, e, 3) for q, e in items), m),
            "Hit@5%": pct(sum(hit_at_k(q, e, 5) for q, e in items), m),
            "MRR": round(sum(mrr(q, e) for q, e in items) / m, 3),
            "误拒数": sum(1 for q, e in items if refused(q)),
        }
    tot = {
        "Hit@1%": pct(sum(hit_at_k(q, e, 1) for _g, q, e in labeled), allq),
        "Hit@3%": pct(sum(hit_at_k(q, e, 3) for _g, q, e in labeled), allq),
        "Hit@5%": pct(sum(hit_at_k(q, e, 5) for _g, q, e in labeled), allq),
        "MRR": round(sum(mrr(q, e) for _g, q, e in labeled) / allq, 3),
        "误拒数": sum(1 for _g, q, e in labeled if refused(q)),
    }

    # ---- 应拒答 ----
    c_rows = [{"query": q, "refused": refused(q),
               "top": [doc_index(d) for d in rag.retrieve(q, k=K_PROD)][:3]} for q in c_list]
    c_ok = sum(1 for x in c_rows if x["refused"])
    z_ok = sum(1 for q in z_list if refused(q))

    # ---- 阈值敏感性：仅纯词法路有意义（混合路的放行判据是名次一致性，不吃 min_score）----
    sweep = []
    if not hybrid:
        for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            rec = sum(1 for _g, q, _e in labeled
                      if rag.retrieve(q, k=K_PROD, min_score=t)) / allq
            ref = sum(1 for q in c_list if not rag.retrieve(q, k=K_PROD, min_score=t)) / len(c_list)
            sweep.append({"min_score": t, "有结果率": round(rec, 3), "正确拒答率": round(ref, 3)})

    # ---- 检索层延迟（不含任何模型调用）----
    lat = []
    pool = [q for _g, q, _e in labeled] + c_list + z_list
    for _ in range(10):
        for q in pool:
            s = time.perf_counter()
            rag.retrieve(q, k=K_PROD, min_score=MIN_SCORE_PROD)
            lat.append((time.perf_counter() - s) * 1000)
    lat.sort()
    q_ = lambda x: round(lat[min(len(lat) - 1, int(len(lat) * x))], 3)  # noqa: E731

    # ---- 随机基线（固定种子，可复现）----
    rnd = random.Random(20260923)
    rb = {"hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0}
    for _g, _q, e in labeled:
        pick = rnd.sample(range(n), 5)
        for kk, key in ((1, "hit1"), (3, "hit3"), (5, "hit5")):
            if e in pick[:kk]:
                rb[key] += 1
        rb["mrr"] += (1.0 / (pick.index(e) + 1)) if e in pick else 0.0

    out = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "被测代码": "app/rag.py::retrieve（生产入口，未重写 BM25）",
        "生效路径": diag["生效路径"],
        "生产参数": {"k": K_PROD, "min_score": MIN_SCORE_PROD,
                     "说明": "min_score=0.0 等价于「BM25 非零分即通过」，并非调优后的相关性阈值；"
                             "启用混合路后放行判据改为两路 top-k 名次一致性"},
        "语料": {"文档数": n, "正文总字数": sum(len(d.page_content) for d in corpus)},
        "查询集": {"文件": "eval/retrieval_queries.json",
                   "应命中": {"A": len(qs["A"]), "B": len(qs["B"]), "合计": allq},
                   "应拒答": {"C": len(c_list), "争议单列": len(z_list)},
                   "标注者": qs.get("_标注者")},
        "分词诊断": diag,
        "指标": {
            "A组_语料内直述": groups["A"],
            "B组_同义改写": groups["B"],
            "合计_应命中": {**tot, "n": allq, "误拒率%": pct(tot["误拒数"], allq)},
            "C组_语料外应拒答": {"n": len(c_list), "正确拒答": c_ok,
                                  "正确拒答率%": pct(c_ok, len(c_list)), "明细": c_rows},
            "争议样本": {"n": len(z_list), "拒答": z_ok, "说明": "产品语义可争议/主要混淆源，不并入 C 主口径"},
        },
        "随机基线_固定种子": {
            "Hit@1%": pct(rb["hit1"], allq), "Hit@3%": pct(rb["hit3"], allq),
            "Hit@5%": pct(rb["hit5"], allq), "MRR": round(rb["mrr"] / allq, 3),
            "理论值": {"Hit@1": "8.3%", "Hit@5": "41.7%"},
        },
        "阈值敏感性": sweep or "（混合路不吃 min_score，本列仅纯词法路输出）",
        "检索层延迟_不含模型调用": {"样本": len(lat), "P50_ms": q_(0.50), "P95_ms": q_(0.95),
                                       "max_ms": round(lat[-1], 3), "建索引_ms": round(build_ms, 1)},
        "局限": ["语料 12 篇，规模远小于生产，绝对值不可外推",
                 "标注由 AI agent（Qoder）生成，非人工、未独立复核",
                 "EMBED_BACKEND=hashing 时语义路无真正语义能力，该列只证明结构正确",
                 "未测真实 Embedding 多路与 RRF 融合质量（需 EMBED_API_KEY）"],
    }
    fp = os.path.join(_ROOT, "eval", "bm25_bench_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    io.open(fp, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))

    print("生效路径: %s" % diag["生效路径"])
    print("语料 %d 篇 | 应命中 %d 条（A%d+B%d）| 应拒答 C=%d 争议=%d" % (
        n, allq, len(qs["A"]), len(qs["B"]), len(c_list), len(z_list)))
    print("分词器 %s → 「%s」切成 %s" % (diag["tokenizer"], sample, diag["tokens"]))
    for g in ("A", "B"):
        d = groups[g]
        print("%s 组 n=%-3d Hit@1/3/5 = %5s%% / %5s%% / %5s%%  MRR %.3f  误拒 %d" % (
            g, d["n"], d["Hit@1%"], d["Hit@3%"], d["Hit@5%"], d["MRR"], d["误拒数"]))
    print("合计      Hit@1/3/5 = %s%% / %s%% / %s%%   MRR %.3f   误拒 %d (%s%%)" % (
        tot["Hit@1%"], tot["Hit@3%"], tot["Hit@5%"], tot["MRR"], tot["误拒数"], pct(tot["误拒数"], allq)))
    print("C 组正确拒答 %d/%d = %s%%   争议 %d/%d" % (c_ok, len(c_list), pct(c_ok, len(c_list)), z_ok, len(z_list)))
    print("随机基线  Hit@1/3/5 = %s%% / %s%% / %s%%   MRR %.3f" % (
        out["随机基线_固定种子"]["Hit@1%"], out["随机基线_固定种子"]["Hit@3%"],
        out["随机基线_固定种子"]["Hit@5%"], out["随机基线_固定种子"]["MRR"]))
    print("延迟 P50 %.3fms / P95 %.3fms / max %.3fms | 建索引 %.1fms" % (
        out["检索层延迟_不含模型调用"]["P50_ms"], out["检索层延迟_不含模型调用"]["P95_ms"],
        out["检索层延迟_不含模型调用"]["max_ms"], out["检索层延迟_不含模型调用"]["建索引_ms"]))
    if sweep:
        print("阈值敏感性:", "  ".join("t=%s→有结果%s%%/拒答%s%%" % (
            s["min_score"], int(s["有结果率"] * 100), int(s["正确拒答率"] * 100)) for s in sweep))
    print("\n报告:", fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
