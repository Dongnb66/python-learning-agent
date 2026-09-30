#!/usr/bin/env python
"""下载本地中文 embedding 模型（BAAI/bge-small-zh-v1.5 的 int8 ONNX）。

为什么要这个脚本
----------------
第二路语义检索要验证「融合到底有没有收益」，就必须用**真语义模型** ——
原来的字符哈希桩的余弦与相关性无关，拿它当第二路等于往 RRF 里按伪随机名次
注入一份与 query 无关的扰动，「双路 Hit@1 没变」因此什么都证明不了。

为什么不用 sentence-transformers
--------------------------------
那要拖 torch（Windows CPU 版约 800 MB）。这里用 int8 ONNX（约 23 MB）
+ onnxruntime（约 15 MB），够用且轻。

为什么要走 hf-mirror.com
-----------------------
huggingface.co 在部分网络下不可达；hf-mirror.com 内容一致且可达。
可用 `HF_ENDPOINT` 覆盖。

用法
----
    python scripts/fetch_embed_model.py                # 默认下到 D:\\Downloads\\models\\bge-small-zh-v1.5
    python scripts/fetch_embed_model.py --dest <目录>
    EMBED_BACKEND=local-onnx EMBED_LOCAL_MODEL_DIR=<目录> python -m pytest -q
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_DEST = r"D:\Downloads\models\bge-small-zh-v1.5"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
REPO = "Xenova/bge-small-zh-v1.5"
FILES = ["onnx/model_int8.onnx", "tokenizer.json", "config.json", "vocab.txt"]


def main() -> int:
    ap = argparse.ArgumentParser(description="下载本地中文 embedding 模型（int8 ONNX）")
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
        if not args.force and os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"  跳过（已存在） {rel}  {os.path.getsize(out) // 1024} KB")
            continue
        try:
            with httpx.stream("GET", base + rel, timeout=300, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = 0
                with open(out, "wb") as fh:
                    for chunk in resp.iter_bytes(65536):
                        fh.write(chunk)
                        total += len(chunk)
            print(f"  ✅ {rel}  {total // 1024} KB")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {rel}  {type(exc).__name__}: {str(exc)[:140]}")

    onnx_ok = any(
        os.path.exists(os.path.join(args.dest, "onnx", n))
        for n in ("model_int8.onnx", "model.onnx")
    )
    tok_ok = os.path.exists(os.path.join(args.dest, "tokenizer.json"))

    print()
    if onnx_ok and tok_ok and not failed:
        print(f"模型就绪：{args.dest}")
        print("启用方式：")
        print(f'  $env:EMBED_BACKEND="local-onnx"; $env:EMBED_LOCAL_MODEL_DIR="{args.dest}"')
        return 0
    print(f"模型不完整（失败 {failed} 个），检索层会自动退回纯词法路径。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
