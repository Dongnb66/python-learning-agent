"""RAG 检索层（防幻觉）。

采用 BM25 关键词检索（langchain_community + rank_bm25），
- 不需要任何 embedding API，离线可用，适合学生本地 Demo 与测试；
- 检索到的片段会作为「事实依据」传给资源智能体，约束 LLM 只基于
  真实存在的资料做推荐，从而降低编造链接/资源的风险。

如需更强语义检索，可把 BM25Retriever 换成 FAISS + Embeddings，
接口保持一致即可。
"""
from __future__ import annotations

import json
import os

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "resources.json")


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


def get_retriever(k: int = 4) -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever.from_documents(_load_documents(), k=k)
    return _retriever


def retrieve(query: str, k: int = 4, min_score: float = 0.0) -> list[Document]:
    """返回与 query 最相关的 k 个资料片段（按 BM25 分数降序）。

    min_score：相关性下限，默认 0.0。
        BM25 分数为 0 表示「query 的词一个都没在语料里命中」——这类结果
        只是凑数的，并不相关。BM25Retriever 原生行为是无论如何都返回
        top-k（含 0 分项），若直接把它们当作 LLM 的「事实依据」，等于
        给模型塞了一堆无关材料，反而助推它编造。
        因此这里做显式分数过滤：资料库确实没有相关内容时返回空列表，
        上层即可据此**在代码层拒答**，而不是指望模型自觉。
    """
    r = get_retriever(k=k)
    scores = r.vectorizer.get_scores(r.preprocess_func(query))
    ranked = sorted(zip(scores, r.docs), key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in ranked[:k] if score > min_score]
