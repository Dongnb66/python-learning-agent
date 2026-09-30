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
- `LocalONNXEmbeddings`：本地跑 `BAAI/bge-small-zh-v1.5` 的 int8 ONNX（约 23 MB），
  `onnxruntime` + `tokenizers`，**不需要 torch**。这一路才是真语义模型。
- `HashingEmbeddings`：字符 bigram 哈希投影，**确定性、零网络、零 Key**。
  它不是语义模型，只用于让混合链路在无 Key 时仍可被单测覆盖 ——
  因此它产出的任何数字都不得当作「检索质量」对外宣称。
- `None`（未配置且未显式要求）：检索层退回纯词法路径，行为与历史版本一致。

⚠️ **哈希桩与真语义路的区别必须说清**：哈希向量的余弦与「相关性」无关，
所以拿它当初二路，等于往 RRF 里按**伪随机名次**注入一份与 query 无关的扰动。
「双路 Hit@1 没变」因此只能证明**这一路没贡献**，不能证明融合本身没收益。
要验证融合，第二路必须是真模型（用 `LocalONNXEmbeddings` 或 OpenAI 兼容端点）。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, Sequence

from app.config import embed_configured, get_settings

_DIM = 512

# BGE 中文模型的 s2p（短查询 → 长段落）检索指令。
# 官方模型卡要求：检索场景下**只给 query 加**，文档不加；给文档也加反而掉分。
_BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class EmbeddingProvider(Protocol):
    """把文本批量映射为等长向量。

    `is_query=True` 表示这批文本是检索查询（而非语料片段）——
    只有需要「查询/文档非对称」处理的模型（如 BGE 的检索指令）才会用到它。
    """

    name: str

    def embed(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]: ...


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

    def embed(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
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

    def embed(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
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


class LocalONNXEmbeddings:
    """本地中文语义 embedding：`BAAI/bge-small-zh-v1.5` 的 int8 ONNX。

    为什么用 ONNX 而不是 sentence-transformers：后者要拖 torch（Windows CPU 版
    约 800 MB），而 int8 ONNX 只要 23 MB + onnxruntime（约 15 MB）。
    模型与分词器由 `scripts/fetch_embed_model.py` 从 hf-mirror.com 拉取
    （huggingface.co 在部分网络下不可达，镜像可达且内容一致）。

    池化用带 attention mask 的 mean pooling，再做 L2 归一化 —— 与 BGE 官方一致。
    """

    name = "local-onnx-bge-small-zh"

    def __init__(self, model_dir: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cand = [
            os.path.join(model_dir, "onnx", "model_int8.onnx"),
            os.path.join(model_dir, "onnx", "model.onnx"),
        ]
        onnx_path = next((p for p in cand if os.path.exists(p)), None)
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if onnx_path is None or not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"本地 embedding 模型不完整：需要 {model_dir}\\onnx\\model_int8.onnx 与 tokenizer.json。"
                " 可运行 scripts/fetch_embed_model.py 自动下载。"
            )

        self.model_dir = model_dir
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=512)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def embed(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
        import numpy as np

        if not texts:
            return []
        payload = [
            (_BGE_ZH_QUERY_INSTRUCTION + t) if is_query else t for t in texts
        ]
        encs = self._tok.encode_batch(list(payload))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed: dict[str, object] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        hidden = self._sess.run(None, feed)[0]          # (B, T, H)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norm, 1e-9, None)).tolist()


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
    if backend in ("local-onnx", "onnx", "local"):
        # 本地真语义模型：不需要 Key、不联网，但需要先下模型
        model_dir = os.getenv("EMBED_LOCAL_MODEL_DIR", "").strip()
        if not model_dir:
            print("[EMBED] EMBED_BACKEND=local-onnx 但未设 EMBED_LOCAL_MODEL_DIR，退回词法路")
            _provider = None
        else:
            try:
                _provider = LocalONNXEmbeddings(model_dir)
                print(f"[EMBED] 已加载本地语义模型：{_provider.name} @ {model_dir}")
            except Exception as exc:
                # 模型缺失/加载失败一律降级，绝不让检索层因此起不来
                print(f"[EMBED] 本地 ONNX 模型加载失败，退回纯词法检索：{exc}")
                _provider = None
    elif backend == "hashing":
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
