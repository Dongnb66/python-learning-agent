"""Embedding 提供者 —— 检索层的第二条路（语义路）。

为什么单独一层
--------------
`app/rag.py` 的 BM25 是词法路：它只能看见字面 token。离线基准实测证明，
在中文语料上「写 HTTP 接口的 Python 框架推荐」与「量子计算基础」会拿到
**完全相同的 BM25 分数并命中同一篇文档**，而前者该命中、后者该拒答 ——
也就是说单靠词法分数无法同时满足「召回」与「空命中拒答」。补一条语义路、
用两路是否一致来决定放不放行，才是能讲清楚的解法。

三种提供者
----------
- `OpenAICompatEmbeddings`：任意 OpenAI 协议兼容的 /v1/embeddings 端点
  （百炼 text-embedding-v3、OpenAI text-embedding-3-small 等）。
- `HashingEmbeddings`：字符 bigram 哈希投影，**确定性、零网络、零 Key**。
  它不是语义模型，只用于让混合链路在无 Key 时仍可被单测覆盖 ——
  因此它产出的任何数字都不得当作「检索质量」对外宣称。
- `None`（未配置且未显式要求）：检索层退回纯词法路径，行为与历史版本一致。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, Sequence

from app.config import embed_configured, get_settings

_DIM = 512


class EmbeddingProvider(Protocol):
    """把文本批量映射为等长向量。"""

    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return 0.0 if na == 0 or nb == 0 else dot / math.sqrt(na * nb)


class HashingEmbeddings:
    """字符 bigram 哈希投影。确定性、无依赖；**不具备语义能力**。"""

    name = "hashing-bigram"

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    @staticmethod
    def _grams(text: str) -> set[str]:
        low = text.lower()
        ascii_words = re.findall(r"[a-z0-9]+", low)
        cjk_runs = re.findall(r"[一-鿿]+", low)
        grams = set(ascii_words)
        for run in cjk_runs:
            if len(run) == 1:
                grams.add(run)
            grams.update(run[i:i + 2] for i in range(len(run) - 1))
        return grams

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            for g in self._grams(t):
                h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class OpenAICompatEmbeddings:
    """OpenAI 协议兼容的 embeddings 端点。"""

    name = "openai-compat"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx  # 延迟导入：离线单测不需要网络依赖

        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in payload]


_provider: EmbeddingProvider | None = None
_resolved = False


def get_embedding_provider() -> EmbeddingProvider | None:
    """返回当前生效的 embedding 提供者；返回 None 表示检索层只走词法路。

    优先级：`EMBED_BACKEND=hashing` 显式指定 → 哈希提供者；
    配置了可用 Key → OpenAI 兼容提供者；否则 None（退回历史行为）。
    """
    global _provider, _resolved
    if _resolved:
        return _provider
    backend = os.getenv("EMBED_BACKEND", "").strip().lower()
    if backend == "hashing":
        _provider = HashingEmbeddings()
    elif backend == "none":
        _provider = None            # 显式关闭语义路：单测与 CI 用它把行为钉在词法降级
    elif not embed_configured():
        _provider = None
    else:
        s = get_settings()
        _provider = OpenAICompatEmbeddings(s.embed_api_key, s.embed_base_url, s.embed_model)
    _resolved = True
    return _provider


def reset_embedding_provider_cache() -> None:
    """清掉提供者单例 —— 供测试与配置热切换使用。"""
    global _provider, _resolved
    _provider = None
    _resolved = False
