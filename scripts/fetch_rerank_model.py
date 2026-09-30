#!/usr/bin/env python
"""下载本地跨编码器重排序模型（`BAAI/bge-reranker-v2-m3` 的 int8 ONNX）。

为什么需要它
------------
《深入理解 AI Agent》第 3 章 L398：

> 「流水线的第三个阶段**神经重排序**并不是为了『补救 RRF 丢掉的得分』才存在的：
> **无论前一步用哪种方式融合，重排序都值得加**，因为它换用了一种更强的匹配范式。」

它同时解决本项目两个问题：
1. 融合产物本该是**并集**（书里 L396「统一的候选池」），交集会丢单路独有命中；
2. 拒答需要**跨查询可比的绝对分**，而 BM25 分与余弦都不可比。

为什么是这个模型
----------------
书里 L400 明确点名 `BAAI/bge-reranker-v2-m3`，且它是**多语**模型（中文可用）。
int8 ONNX 约 544 MB；不用 sentence-transformers 是为了避开 torch（约 800 MB）。

用法
----
    python scripts/fetch_rerank_model.py
    python scripts/fetch_rerank_model.py --dest <目录> --endpoint <镜像>

模型**不进仓库**（544 MB），也不进默认路径 —— 检索在没有它时保持历史行为。
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_DEST = r"D:\Downloads\models\bge-reranker-v2-m3"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
FILES = ["onnx/model_int8.onnx", "tokenizer.json", "config.json"]

# 每个文件的最小可接受大小（字节）。用来识破「上次下到一半就断了」的半成品 ——
# 只判「文件存在」是不够的：中断过一次的 32 MB 残留会被当成已完成，
# 然后在加载时报一个和下载完全无关的错。
MIN_SIZE = {
    "onnx/model_int8.onnx": 500 * 1024 * 1024,   # 实测约 544 MB
    "tokenizer.json": 10 * 1024 * 1024,           # 实测约 16 MB
    "config.json": 100,                           # 实测约 850 B
}


def _complete(path: str, rel: str) -> bool:
    """文件存在**且**大小达到预期下限，才算完整。"""
    return os.path.exists(path) and os.path.getsize(path) >= MIN_SIZE.get(rel, 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description="下载本地重排序模型（int8 ONNX）")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"目标目录（默认 {DEFAULT_DEST}）")
    ap.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    args = ap.parse_args()

    try:
        import httpx
    except ImportError:
        print("需要 httpx：pip install httpx", file=sys.stderr)
        return 2

    base = f"{args.endpoint.rstrip('/')}/{REPO}/resolve/main/"
    os.makedirs(args.dest, exist_ok=True)
    failed = 0

    for rel in FILES:
        out = os.path.join(args.dest, *rel.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not args.force and _complete(out, rel):
            print(f"  跳过（已完整） {rel}  {os.path.getsize(out) // 1024 // 1024} MB")
            continue
        if os.path.exists(out):
            print(f"  ⚠️ {rel} 已有 {os.path.getsize(out) // 1024 // 1024} MB 但未达下限，重下")
        # 先写 .part 再改名：中断时不会留下一个「看起来完整」的文件
        part = out + ".part"
        try:
            with httpx.stream("GET", base + rel, timeout=1800, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = 0
                with open(part, "wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
                        total += len(chunk)
                        if total % (100 << 20) < (1 << 20):
                            print(f"    {rel}: {total // 1024 // 1024} MB ...", flush=True)
            if not _complete(part, rel):
                raise IOError(f"下载不完整：{total} 字节 < 下限 {MIN_SIZE.get(rel)}")
            os.replace(part, out)
            print(f"  ✅ {rel}  {total // 1024 // 1024} MB")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {rel}  {type(exc).__name__}: {str(exc)[:140]}")
            if os.path.exists(part):
                os.remove(part)          # 清掉半成品，避免下次又被误判

    onnx_ok = any(
        os.path.exists(os.path.join(args.dest, "onnx", n))
        for n in ("model_int8.onnx", "model_quantized.onnx", "model.onnx")
    )
    tok_ok = os.path.exists(os.path.join(args.dest, "tokenizer.json"))

    print()
    if onnx_ok and tok_ok and not failed:
        print(f"重排序模型就绪：{args.dest}")
        print("启用方式（与 embedding 一起）：")
        print(f'  $env:EMBED_BACKEND="local-onnx"; $env:EMBED_LOCAL_MODEL_DIR="D:\\Downloads\\models\\bge-small-zh-v1.5"')
        print(f'  $env:RERANK_BACKEND="local-onnx"; $env:RERANK_LOCAL_MODEL_DIR="{args.dest}"')
        return 0
    print(f"模型不完整（失败 {failed} 个），检索会自动跳过重排序。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
