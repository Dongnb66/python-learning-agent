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


def retrieve(query: str, k: int = 4) -> list[Document]:
    """返回与 query 最相关的 k 个资料片段。"""
    return get_retriever(k=k).invoke(query)
