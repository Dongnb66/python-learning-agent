# -*- coding: utf-8 -*-
"""验证简历头条「Hit@1 22.2% → 77.8%」是否统计显著。

为什么单独验这一条
------------------
这一条和「混合检索让 Hit@1 +2.8pt」是**两种不同性质**的结论：

* 22.2% → 77.8% 是 **CJK bigram 分词**带来的，**20 个百分点**；
* 77.8% → 80.6% 是混合检索带来的，**2.8 个百分点**，已验过 **不显著**。

不能因为后者不显著就把前者也一起否掉 —— 幅度差 7 倍。
所以这里用同一套配对方法（McNemar 精确检验 + 配对 bootstrap）**单独验头条**。

⚠️ 本脚本**不碰生产路径**：直接构造两种分词器的 BM25Retriever 对比，
不改 `app/rag.py` 的默认行为。

用法：
    python scripts/tokenizer_significance.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import rag  # noqa: E402

K_PROD = 4
MIN_SCORE = 2.0
BOOT = 10000


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def paired_bootstrap(a: list[bool], b: list[bool], iters: int = BOOT, seed: int = 20260930):
    rng = random.Random(seed)
    n = len(a)
    deltas = [int(x) - int(y) for x, y in zip(a, b)]
    obs = sum(deltas) / n
    s = []
    for _ in range(iters):
        t = 0
        for _ in range(n):
            t += deltas[rng.randrange(n)]
        s.append(t / n)
    s.sort()
    lo, hi = s[int(0.025 * iters)], s[int(0.975 * iters)]
    left = sum(1 for x in s if x <= 0) / iters
    right = sum(1 for x in s if x >= 0) / iters
    return obs, lo, hi, min(1.0, 2 * min(left, right))


def main() -> int:
    corpus = rag._load_documents()
    url2idx = {d.metadata.get("url", ""): i for i, d in enumerate(corpus)}
    qs = json.load(open(os.path.join(_ROOT, "eval", "retrieval_queries.json"), encoding="utf-8"))
    labeled = [(x["q"], x["expected"]) for x in qs["A"]] + [(x["q"], x["expected"]) for x in qs["B"]]
    n = len(labeled)

    blank = rag.get_retriever(k=K_PROD)          # 历史默认分词（空白切分）
    bigram = rag._get_bigram_retriever(k=K_PROD)  # CJK bigram

    def hits(retriever) -> list[bool]:
        out = []
        for q, exp in labeled:
            scores = rag._lexical_scores(retriever, q)
            ranked = sorted(enumerate(scores), key=lambda p: -p[1])
            keeps = [(i, s) for i, s in ranked if s > MIN_SCORE][:K_PROD]
            out.append([i for i, _ in keeps][:1] == [exp])
        return out

    h_blank, h_bigram = hits(blank), hits(bigram)
    kb, kg = sum(h_blank), sum(h_bigram)

    print(f"n = {n} 有答案查询，纯词法路（min_score={MIN_SCORE}）")
    print()
    print(f"{'分词器':<28}{'命中':>5}{'Hit@1':>9}")
    print(f"{'空白切分（历史默认）':<28}{kb:>5}{100*kb/n:>8.1f}%")
    print(f"{'CJK bigram（现在）':<28}{kg:>5}{100*kg/n:>8.1f}%")
    print(f"  => 简历头条写的 22.2% -> 77.8%  " +
          ("✅ 与本脚本一致" if abs(100*kb/n - 22.2) < 1.5 and abs(100*kg/n - 77.8) < 1.5
           else f"⚠️ 与实测不符（本脚本 {100*kb/n:.1f}% -> {100*kg/n:.1f}%）"))
    print()

    b = sum(1 for x, y in zip(h_bigram, h_blank) if x and not y)   # bigram 对、blank 错
    c = sum(1 for x, y in zip(h_bigram, h_blank) if y and not x)   # bigram 错、blank 对
    p_mc = mcnemar_exact(b, c)
    delta, lo, hi, p_bs = paired_bootstrap(h_bigram, h_blank)
    print(f"配对分析（bigram vs 空白切分）：")
    print(f"  不一致对：仅 bigram 对 = {b}   仅空白对 = {c}")
    print(f"  McNemar 精确检验 p = {p_mc:.3e}  ->  {'✅ 显著' if p_mc < 0.05 else '❌ 不显著'}")
    print(f"  配对 bootstrap 差值 {100*delta:+.1f}pt，95%CI = [{100*lo:+.1f}, {100*hi:+.1f}]pt")
    print()
    se = math.sqrt((kg/n) * (1 - kg/n) / n)
    print(f"对照：n={n} 时 95% 置信区间约 ±{100*1.96*se:.1f}pt（正态近似）")
    print()
    print("结论：")
    if p_mc < 0.05:
        print("  ✅ 简历头条「22.2% -> 77.8%」统计显著 —— 这条可以照写。")
        print("  ⚠️ 但「混合检索额外 +2.8pt」不显著（见 scripts/significance_probe.py）——")
        print("     两个结论要分开说，不要因为后者不显著就把头条也一起否定。")
    else:
        print("  ❌ 头条本身也不显著 —— 需要扩大评估集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
