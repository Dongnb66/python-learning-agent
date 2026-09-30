# -*- coding: utf-8 -*-
"""把「真 embedding vs 哈希桩 vs 纯词法」的三方对照写成可复现的存档。

背景：2026-09-30 把第二路的字符哈希桩换成真语义模型（bge-small-zh-v1.5 int8 ONNX），
重跑离线基准，得到一个和此前口径不同的结论 —— 同时修掉了基准脚本里一个真 bug。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PY = sys.executable
CONFIGS = [
    ("lexical", {"EMBED_BACKEND": "none"}, "纯词法（bigram BM25 + 阈值）"),
    ("hybrid-hashing", {"EMBED_BACKEND": "hashing"}, "混合 · 哈希桩（无真语义能力）"),
    ("hybrid-onnx", {"EMBED_BACKEND": "local-onnx",
                     "EMBED_LOCAL_MODEL_DIR": r"D:\Downloads\models\bge-small-zh-v1.5"},
     "混合 · 真语义（bge-small-zh-v1.5 int8 ONNX）"),
]


def run(env_over: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for k in ("EMBED_BACKEND", "EMBED_LOCAL_MODEL_DIR"):
        env.pop(k, None)
    env.update(env_over)
    # 基准脚本每次都写 eval/bm25_bench_<ts>.json，这里取运行后最新的那个
    before = set(os.listdir(os.path.join(_ROOT, "eval")))
    p = subprocess.run([PY, os.path.join(_ROOT, "scripts", "bm25_offline_bench.py")],
                       cwd=_ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout or "")[-800:]}
    after = set(os.listdir(os.path.join(_ROOT, "eval")))
    new = sorted(after - before, key=lambda f: os.path.getmtime(os.path.join(_ROOT, "eval", f)))
    for f in reversed(new):
        if f.startswith("bm25_bench_") and f.endswith(".json"):
            return json.load(open(os.path.join(_ROOT, "eval", f), encoding="utf-8"))
    return {"error": "未找到新生成的基准 JSON"}


def brief(d: dict) -> dict:
    if "error" in d:
        return d
    g = d.get("分组") or d.get("groups") or {}
    return d


def main() -> int:
    out = {"生成时间": datetime.now().isoformat(timespec="seconds"),
           "目的": "把第二路从字符哈希桩换成真语义模型，重跑离线基准，看融合结论是否改变",
           "被测代码": "app/rag.py::retrieve（生产入口）",
           "查询集": "eval/retrieval_queries.json（60 条：A12 / B24 / C22 / C_争议2）",
           "⚠️ 期间修掉的基准 bug":
               "hit_at_k 原来传 rag.retrieve(query, k=k)。纯词法下 k 只是截断，"
               "但混合路径下 k 会改变算法（放行条件 agree = lex_top ∩ sem_top）——"
               "传 k=1 等于要求同一篇文档既是词法第一又是语义第一。"
               "症状：真模型下 Hit@1 被算成 52.8%，同一次检索按生产 k=4 取首条是 80.6%。"
               "已改为「用生产 k 检索、再按需要的 k 截断」。",
           "结果": {}}

    for key, env_over, label in CONFIGS:
        print(f"跑 {label} ...", flush=True)
        d = run(env_over)
        if "error" in d:
            print("  ❌", d["error"][:200])
            out["结果"][key] = d
            continue
        out["结果"][key] = {"说明": label, "报告": d}
        print("  ✅")

    dest = os.path.join(_ROOT, "eval", "fusion_real_embedding_20260930.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n已写出 {dest}  ({os.path.getsize(dest)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
