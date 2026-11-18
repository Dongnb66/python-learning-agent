# -*- coding: utf-8 -*-
"""从真实后端生成 BM25 对拍夹具（零 Key、纯词法路，与 tests/conftest.py 同口径）。

用法（仓库根目录）：.venv/Scripts/python.exe playground/tests/make_fixtures.py
输出：playground/tests/fixtures/bm25_expected.json
"""
from __future__ import annotations

import json
import os
import sys

os.environ["EMBED_BACKEND"] = "none"
os.environ["LLM_API_KEY"] = ""
os.environ["EMBED_API_KEY"] = ""

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app import rag  # noqa: E402

QUERIES = [
    "学习 LangGraph",
    "摩托车发动机化油器怎么清洗",
    "我想学点编程",
    "怎么用 Docker 部署 Python 应用",
    "RAG 防幻觉",
    "乳腺癌的靶向治疗方案有哪些",
    "FastAPI",
    "递归 指针",
    "Python 与 FastAPI 基础巩固",
    "LangChain / LangGraph 多智能体入门",
    "RAG 检索增强实践",
    "动手做一个校园场景 Agent",
    "部署与工程化（Docker）",
    "算法与数据结构需补齐",
    "工程化部署经验不足",
    "学习方法",
]

TOKEN_SAMPLES = [
    "学习 LangGraph",
    "怎么用 Docker 部署 Python 应用",
    "Python3.12 是 good 的！",
    "单个字",
    "混合ABC中文def123",
]


def main() -> None:
    rag.reset_retriever_cache()
    r = rag._get_bigram_retriever(k=4)
    docs = rag._all_docs()
    out = {
        "corpus_titles": [d.metadata["title"] for d in docs],
        "tokenize": {t: rag.tokenize(t) for t in TOKEN_SAMPLES},
        "queries": {},
    }
    for q in QUERIES:
        scores = [float(s) for s in rag._lexical_scores(r, q)]
        hits = rag.retrieve(q, k=4, min_score=2.0)
        out["queries"][q] = {
            "scores": scores,
            "hits": [
                {"title": d.metadata["title"], "url": d.metadata["url"], "type": d.metadata["type"]}
                for d in hits
            ],
        }
    dest = os.path.join(os.path.dirname(__file__), "fixtures", "bm25_expected.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("fixtures written:", dest)


if __name__ == "__main__":
    main()