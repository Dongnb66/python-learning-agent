# -*- coding: utf-8 -*-
"""BM25 检索层离线基准（零 API Key、零网络、零 LLM）。

为什么能离线：BM25 是确定性算法，检索层本身不调模型，所以命中率与延迟
都是**真实测量值**，不是 mock 出来的 —— 与 run_eval.py 的端到端评测互补。

⚠️ 三条必须一起读的局限（否则数字会被误读）：
  1. 语料仅 12 篇 / 849 字。单相关文档下随机基线 Hit@5 = 5/12 ≈ 41.7%，
     所以只报绝对命中率没有意义，本脚本同时报随机基线与相对提升。
  2. 查询与「应命中文档」的标注由本脚本作者给出，非独立标注者，属弱监督。
  3. 本脚本测的是**检索层排序行为**，不是真实用户满意度。

用法：python scripts/bm25_offline_bench.py
输出：终端表格 + eval/bm25_bench_<时间戳>.json
"""
from __future__ import annotations

import io
import json
import os
import random
import statistics
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

# ---- 查询集：A 组=语料内直述，B 组=语料内但词面不匹配（同义改写），C 组=语料外应拒答 ----
QUERIES_A = [
    ("Python 变量和循环怎么入门", 0),
    ("FastAPI 怎么写接口", 1),
    ("LangChain 的模型和提示怎么用", 2),
    ("LangGraph 怎么做多智能体编排", 3),
    ("RAG 检索增强生成的原理", 4),
    ("SQLAlchemy 2.0 怎么上手", 5),
    ("Python 应用怎么容器化部署", 6),
    ("提示词工程有什么技巧", 7),
    ("数据结构和算法怎么刷题", 8),
    ("HTTP 和 TCP 的核心概念", 9),
    ("SQL 索引和查询语法", 10),
    ("机器学习回归分类入门", 11),
]
QUERIES_B = [
    ("我想学后端框架，写 REST 接口", 1),
    ("大模型的 prompt 调优方法", 7),
    ("怎么把项目打包成镜像跑起来", 6),
    ("向量检索和召回相关度", 4),
    ("关系型数据库的 ORM 对象映射", 5),
    ("不会算法，想从链表和哈希表开始补", 8),
    ("网络协议搞不懂，比如三次握手", 9),
    ("监督学习里的预测任务怎么入门", 11),
]
QUERIES_C = [
    "Kubernetes 的 etcd 集群脑裂怎么处理",
    "如何申请美国的 EB-5 投资移民签证",
    "乳腺癌的靶向治疗方案有哪些",
    "国债期货的久期对冲怎么做",
    "摩托车发动机化油器怎么清洗",
    "莎士比亚十四行集的韵律分析",
    "热带鱼缸水藻爆发怎么处理",
    "航空发动机涡轮叶片单晶铸造工艺",
]

Labeled = [("A", q, e) for q, e in QUERIES_A] + [("B", q, e) for q, e in QUERIES_B]


def doc_index(doc) -> int:
    """把返回的 Document 映射回语料下标（按 url 对齐）。"""
    return _URL2IDX.get(doc.metadata.get("url", ""), -1)


def hit_at_k(query: str, expected: int, k: int) -> bool:
    return any(doc_index(d) == expected for d in rag.retrieve(query, k=k, min_score=MIN_SCORE_PROD))


def mrr(query: str, expected: int) -> float:
    for rank, d in enumerate(rag.retrieve(query, k=10, min_score=MIN_SCORE_PROD), 1):
        if doc_index(d) == expected:
            return 1.0 / rank
    return 0.0


def main() -> int:
    global _URL2IDX
    corpus = rag._load_documents()
    _URL2IDX = {d.metadata.get("url", ""): i for i, d in enumerate(corpus)}
    n = len(corpus)

    # ---- 分词诊断：报告本次实际生效的那条路 ----
    sample = "Python 变量和循环怎么入门"
    from app.embeddings import get_embedding_provider
    hybrid = get_embedding_provider() is not None
    active = rag._get_bigram_retriever(k=K_PROD) if hybrid else rag.get_retriever(k=K_PROD)
    tokens = active.preprocess_func(sample)
    diag_tokens = {"query": sample, "tokens": list(tokens),
                   "生效路径": "混合（bigram BM25 + Embedding + RRF）" if hybrid else "纯词法（空白分词 BM25）",
                   "tokenizer": getattr(active.preprocess_func, "__name__", "callable")}

    # ---- 预热 + 建索引耗时 ----
    t0 = time.perf_counter()
    rag.retrieve("预热 warmup", k=K_PROD)
    build_ms = (time.perf_counter() - t0) * 1000

    # ---- 主指标 ----
    res = {}
    for grp in ("A", "B"):
        items = [(q, e) for g, q, e in Labeled if g == grp]
        res[grp] = {
            "n": len(items),
            "hit1": sum(hit_at_k(q, e, 1) for q, e in items),
            "hit3": sum(hit_at_k(q, e, 3) for q, e in items),
            "hit5": sum(hit_at_k(q, e, 5) for q, e in items),
            "mrr": sum(mrr(q, e) for q, e in items),
            "empty": sum(1 for q, e in items if not rag.retrieve(q, k=K_PROD, min_score=MIN_SCORE_PROD)),
        }
    allq = res["A"]["n"] + res["B"]["n"]
    tot = {k: res["A"][k] + res["B"][k] for k in ("hit1", "hit3", "hit5", "mrr", "empty")}

    # ---- C 组：应拒答 ----
    c_rows = []
    for q in QUERIES_C:
        got = rag.retrieve(q, k=K_PROD, min_score=MIN_SCORE_PROD)
        c_rows.append({"query": q, "refused": not got,
                       "top": [doc_index(d) for d in got[:3]]})
    c_ok = sum(1 for x in c_rows if x["refused"])

    # ---- 阈值敏感性（生产阈值是 >0，这里扫更高阈值看代价）----
    sweep = []
    for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        recall = sum(1 for _g, q, _e in Labeled if rag.retrieve(q, k=K_PROD, min_score=t)) / allq
        refuse = sum(1 for q in QUERIES_C if not rag.retrieve(q, k=K_PROD, min_score=t)) / len(QUERIES_C)
        sweep.append({"min_score": t, "有结果率": round(recall, 3), "正确拒答率": round(refuse, 3)})

    # ---- 延迟：检索层单次查询，不含任何模型调用 ----
    lat = []
    pool = [q for _g, q, _e in Labeled] + QUERIES_C
    for _ in range(20):
        for q in pool:
            s = time.perf_counter()
            rag.retrieve(q, k=K_PROD, min_score=MIN_SCORE_PROD)
            lat.append((time.perf_counter() - s) * 1000)
    lat.sort()
    p = lambda x: round(lat[min(len(lat) - 1, int(len(lat) * x))], 3)  # noqa: E731

    # ---- 随机基线（固定种子，可复现）----
    rnd = random.Random(20260923)
    rb = {"hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0}
    for _g, q, e in Labeled:
        pick = rnd.sample(range(n), 5)
        for kk, key in ((1, "hit1"), (3, "hit3"), (5, "hit5")):
            if e in pick[:kk]:
                rb[key] += 1
        rb["mrr"] += (1.0 / (pick.index(e) + 1)) if e in pick else 0.0

    pct = lambda a, b: round(100.0 * a / b, 1) if b else 0.0  # noqa: E731
    out = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "被测代码": "app/rag.py::retrieve（生产入口，未重写 BM25）",
        "生产参数": {"k": K_PROD, "min_score": MIN_SCORE_PROD,
                     "说明": "min_score=0.0 等价于「BM25 非零分即通过」，并非调优后的相关性阈值"},
        "语料": {"文档数": n, "正文总字数": sum(len(d.page_content) for d in corpus)},
        "分词诊断": diag_tokens,
        "指标": {
            "A组_语料内直述": {**res["A"], "Hit@1%": pct(res["A"]["hit1"], res["A"]["n"]),
                                "Hit@3%": pct(res["A"]["hit3"], res["A"]["n"]),
                                "Hit@5%": pct(res["A"]["hit5"], res["A"]["n"]),
                                "MRR": round(res["A"]["mrr"] / res["A"]["n"], 3),
                                "空返回数": res["A"]["empty"]},
            "B组_同义改写": {**res["B"], "Hit@1%": pct(res["B"]["hit1"], res["B"]["n"]),
                              "Hit@5%": pct(res["B"]["hit5"], res["B"]["n"]),
                              "MRR": round(res["B"]["mrr"] / res["B"]["n"], 3),
                              "空返回数": res["B"]["empty"]},
            "合计_20条标注查询": {"Hit@1%": pct(tot["hit1"], allq), "Hit@3%": pct(tot["hit3"], allq),
                                  "Hit@5%": pct(tot["hit5"], allq), "MRR": round(tot["mrr"] / allq, 3),
                                  "误拒数(应命中却空返回)": tot["empty"]},
            "C组_语料外应拒答": {"n": len(QUERIES_C), "正确拒答": c_ok,
                                  "正确拒答率%": pct(c_ok, len(QUERIES_C)),
                                  "明细": c_rows},
        },
        "随机基线_12篇语料单相关文档": {
            "Hit@1%": pct(rb["hit1"], allq), "Hit@3%": pct(rb["hit3"], allq),
            "Hit@5%": pct(rb["hit5"], allq), "MRR": round(rb["mrr"] / allq, 3),
            "理论值": {"Hit@1": "8.3%", "Hit@5": "41.7%"},
        },
        "阈值敏感性": sweep,
        "检索层延迟_不含模型调用": {"样本": len(lat), "P50_ms": p(0.50), "P95_ms": p(0.95),
                                       "max_ms": round(lat[-1], 3), "建索引_ms": round(build_ms, 1)},
        "局限": ["语料 12 篇，规模远小于生产，绝对值不可外推",
                 "标注由脚本作者给出，非独立标注",
                 "未测 Embedding 多路与 RRF 融合（该路径需要 API Key）"],
    }
    os.makedirs(os.path.join(_ROOT, "eval"), exist_ok=True)
    fp = os.path.join(_ROOT, "eval", "bm25_bench_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    io.open(fp, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))

    m = out["指标"]["合计_20条标注查询"]
    print("语料 %d 篇 | 标注查询 %d 条 | 生产参数 k=%d min_score=%s" % (n, allq, K_PROD, MIN_SCORE_PROD))
    print("分词器: %s → 「%s」切成 %s" % (diag_tokens["tokenizer"], sample, diag_tokens["tokens"]))
    print("A 组 Hit@1/3/5 = %s%% / %s%% / %s%%   MRR %.3f" % (
        pct(res['A']['hit1'], 12), pct(res['A']['hit3'], 12), pct(res['A']['hit5'], 12), res['A']['mrr'] / 12))
    print("B 组 Hit@1/5   = %s%% / %s%%          MRR %.3f   空返回 %d/%d" % (
        pct(res['B']['hit1'], 8), pct(res['B']['hit5'], 8), res['B']['mrr'] / 8, res['B']['empty'], 8))
    print("合计 Hit@1/3/5 = %s%% / %s%% / %s%%   MRR %.3f   误拒 %d 条" % (
        m["Hit@1%"], m["Hit@3%"], m["Hit@5%"], m["MRR"], m["误拒数(应命中却空返回)"]))
    print("C 组正确拒答 %d/%d = %s%%" % (c_ok, len(QUERIES_C), pct(c_ok, len(QUERIES_C))))
    print("随机基线 Hit@1/3/5 = %s%% / %s%% / %s%%  MRR %.3f" % (
        out["随机基线_12篇语料单相关文档"]["Hit@1%"], out["随机基线_12篇语料单相关文档"]["Hit@3%"],
        out["随机基线_12篇语料单相关文档"]["Hit@5%"], out["随机基线_12篇语料单相关文档"]["MRR"]))
    print("延迟 P50 %.3fms / P95 %.3fms / max %.3fms | 建索引 %.1fms" % (
        out["检索层延迟_不含模型调用"]["P50_ms"], out["检索层延迟_不含模型调用"]["P95_ms"],
        out["检索层延迟_不含模型调用"]["max_ms"], out["检索层延迟_不含模型调用"]["建索引_ms"]))
    print("阈值敏感性:", " ".join("t=%s→有结果%s%%/拒答%s%%" % (
        s["min_score"], int(s["有结果率"] * 100), int(s["正确拒答率"] * 100)) for s in sweep))
    print("\n报告:", fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
