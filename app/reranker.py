"""本地 ONNX 跨编码器重排序器（`BAAI/bge-reranker-v2-m3` int8）。

为什么需要它
------------
《深入理解 AI Agent》第 3 章 L398 逐字：

> 「流水线的第三个阶段**神经重排序（Neural Reranking）** 并不是为了
> 『补救 RRF 丢掉的得分』才存在的：**无论前一步用哪种方式融合，重排序都值得加**，
> 因为它换用了一种更强的匹配范式。它让跨编码器对查询和文档做深度交互匹配，
> 精度远高于检索阶段双编码器各自独立编码、再靠向量运算比相似度的做法。」

它同时解决本仓库两个具体问题：

1. **融合产物应该是并集。** 书里 L396 说融合产出「统一的候选池」，
   而 `app/rag.py` 原来用 `agree = lex_top ∩ sem_top`（交集），
   会丢掉「只有一路能召回」的文档 —— 实测并集命中 31 条、交集只有 29 条（n=36）。
2. **拒答需要跨查询可比的绝对分。** BM25 分和余弦相似度都跨查询/跨模型不可比
   （本仓库实测过：换成哈希向量后 20/20 全误拒）。跨编码器输出的是**绝对相关性分**，
   可以直接当阈值 —— 这正是书里「并集候选池 + 重排序精排 + 绝对分拒答」的标准做法。

为什么用 ONNX 而不是 sentence-transformers
------------------------------------------
同上：后者要拖 torch（Windows CPU 版约 800 MB）。int8 ONNX 约 544 MB + onnxruntime。
模型由 `scripts/fetch_rerank_model.py` 从 hf-mirror.com 拉取。
"""
from __future__ import annotations

import math
import os
from typing import Sequence


class LocalONNXReranker:
    """跨编码器重排序：给 (query, doc) 对打分，分越高越相关。

    `bge-reranker-v2-m3` 是**多语**模型（书里 L400 点名推荐），中文可用。
    """

    name = "local-onnx-bge-reranker-v2-m3"

    def __init__(self, model_dir: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cand = [
            os.path.join(model_dir, "onnx", "model_int8.onnx"),
            os.path.join(model_dir, "onnx", "model_quantized.onnx"),
            os.path.join(model_dir, "onnx", "model.onnx"),
        ]
        onnx_path = next((p for p in cand if os.path.exists(p)), None)
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if onnx_path is None or not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"重排序模型不完整：需要 {model_dir}\\onnx\\model_int8.onnx 与 tokenizer.json。"
                " 可运行 scripts/fetch_rerank_model.py 自动下载。"
            )

        self.model_dir = model_dir
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=512)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """返回每个文档的相关性分（sigmoid 之后的 0–1）。

        跨编码器把 query 与 doc **拼成一段**送进模型逐词比对，
        所以这里必须成对编码（`encode_batch` 传元组），不能各自编码再算相似度。
        """
        import numpy as np

        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        encs = self._tok.encode_batch(pairs)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed: dict[str, object] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        out = self._sess.run(None, feed)[0]          # (B, 1) 或 (B,)
        logits = np.asarray(out).reshape(len(pairs), -1)[:, 0]
        # v2 系列用 sigmoid 把 logit 压到 0–1；官方 compute_score 同此
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


_reranker: LocalONNXReranker | None = None
_resolved = False


def get_reranker():
    """返回当前生效的重排序器；None 表示不做重排序（保持历史行为）。

    默认**关闭** —— 保持「clone 即可跑测试」。显式开启：

        EMBED_BACKEND=local-onnx        EMBED_LOCAL_MODEL_DIR=<embed 模型>
        RERANK_BACKEND=local-onnx       RERANK_LOCAL_MODEL_DIR=<rerank 模型>
    """
    global _reranker, _resolved
    if _resolved:
        return _reranker
    backend = os.getenv("RERANK_BACKEND", "").strip().lower()
    if backend in ("local-onnx", "onnx", "local"):
        model_dir = os.getenv("RERANK_LOCAL_MODEL_DIR", "").strip()
        if not model_dir:
            print("[RERANK] RERANK_BACKEND=local-onnx 但未设 RERANK_LOCAL_MODEL_DIR，跳过重排序")
            _reranker = None
        else:
            try:
                _reranker = LocalONNXReranker(model_dir)
                print(f"[RERANK] 已加载重排序模型：{_reranker.name} @ {model_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[RERANK] 重排序模型加载失败，跳过重排序：{exc}")
                _reranker = None
    else:
        _reranker = None
    _resolved = True
    return _reranker


def reset_reranker_cache() -> None:
    """清掉单例 —— 供测试与配置热切换使用。"""
    global _reranker, _resolved
    _reranker = None
    _resolved = False
