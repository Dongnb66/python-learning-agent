# -*- coding: utf-8 -*-
"""测试会话夹具：把检索层钉死在「纯词法降级」模式。

为什么必须钉住：`.env` 里有没有 `EMBED_API_KEY` 会改变 `retrieve()` 的放行判据
（词法「零分即空」 vs 混合「两路名次一致性」），于是同一个 commit 在配了密钥的
开发者机器上红、在 CI 上绿 —— 这种结果依赖秘密文件的测试等于没有测试。

守卫契约（语料外话题必须返回空、拒答不得调模型）本来就是针对词法路径写的断言，
把它绑在"本机是否配了 key"上是错的。真实语义路径的质量数字不在单测里断言：
那是 `scripts/bm25_offline_bench.py` 的职责（需要网络与额度），本目录只保证
结构、降级与守卫行为可离线确定复现。
"""
from __future__ import annotations

import pytest

from app import embeddings, rag


@pytest.fixture(autouse=True)
def _pin_lexical_mode(monkeypatch):
    """默认全部走降级路径；需要混合路径的用例在自己内部 setenv 覆盖即可。"""
    monkeypatch.setenv("EMBED_BACKEND", "none")
    embeddings.reset_embedding_provider_cache()
    rag.reset_retriever_cache()
    yield
    embeddings.reset_embedding_provider_cache()
    rag.reset_retriever_cache()
