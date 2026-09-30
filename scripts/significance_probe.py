# -*- coding: utf-8 -*-
"""对检索层三种配置做配对显著性检验（McNemar 精确检验 + 配对 bootstrap）。

为什么必须做这个
----------------
《深入理解 AI Agent》第 7 章「评估结果的统计显著性」：
  「若在 n 个用例上测得成功率 p，标准误 ≈ sqrt(p(1-p)/n)……
    100 个用例、成功率 70% 时，95% 置信区间约为 70%±9 个百分点；
    『新模型 73% 对旧模型 70%』不足以支持切换。」
  「若预期收益只有 2–3 个百分点，而评估集只有几十题，应该先扩大样本。」
  「同一批任务比较两个配置时，应优先做**配对分析**：逐题记录谁胜出，
    用 McNemar 检验或配对 bootstrap 判断差异，而不是直接相减两个独立成功率。」

本项目的处境正是这句话：36 条有答案查询、预期收益约 2.8 个百分点。
所以「混合比纯词法好」这个结论**必须先过配对检验**才能说。

配对检验比「两个独立比例相减」灵敏得多 —— 因为它消掉了题目难度这个共变量：
难题在两个配置下都错，不构成信息；只有「一个配置对、另一个错」的题才计入。

用法：
    python scripts/significance_probe.py
    EMBED_BACKEND=local-onnx EMBED_LOCAL_MODEL_DIR=<目录> python scripts/significance_probe.py
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
from app.embeddings import cosine, get_embedding_provider  # noqa: E402

K_PROD = 4
BOOT = 10000


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间 —— 比正态近似在极端比例下更稳。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def normal_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """书里用的正态近似（SE = sqrt(p(1-p)/n)），用来做对照。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）。

    b = A 对 B 错的不一致对数；c = A 错 B 对的不一致对数。
    在 H0（两配置等价）下 b ~ Binomial(b+c, 0.5)。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(a: list[bool], b: list[bool], iters: int = BOOT, seed: int = 20260930):
    """配对 bootstrap：对「逐题差值」重采样，看差值分布是否跨 0。"""
    rng = random.Random(seed)
    n = len(a)
    deltas = [int(x) - int(y) for x, y in zip(a, b)]
    obs = sum(deltas) / n
    samples = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[int(0.975 * iters)]
    # 双侧 p：差值分布里跨过 0 的比例（用 0 在分布中的位置近似）
    left = sum(1 for x in samples if x <= 0) / iters
    right = sum(1 for x in samples if x >= 0) / iters
    p = min(1.0, 2 * min(left, right))
    return obs, lo, hi, p


def main() -> int:
    corpus = rag._load_documents()
    url2idx = {d.metadata.get("url", ""): i for i, d in enumerate(corpus)}
    qs = json.load(open(os.path.join(_ROOT, "eval", "retrieval_queries.json"), encoding="utf-8"))
    labeled = [(x["q"], x["expected"]) for x in qs["A"]] + [(x["q"], x["expected"]) for x in qs["B"]]
    n = len(labeled)

    provider = get_embedding_provider()
    bigram = rag._get_bigram_retriever(k=K_PROD)
    doc_vecs = provider.embed([d.page_content for d in corpus]) if provider else None

    def idx_of(docs):
        return [url2idx.get(d.metadata.get("url", ""), -1) for d in docs]

    def run_lexical(q):
        return idx_of(rag._legacy_retrieve(q, K_PROD, 2.0))

    def run_prod(q):                      # 生产混合路径（交集 + RRF）
        if doc_vecs is None:
            return run_lexical(q)
        r = rag._hybrid_retrieve(q, K_PROD)
        return [] if r is None else idx_of(r)

    def run_union(q):                     # 并集 + RRF（去掉一致性过滤）
        if doc_vecs is None:
            return run_lexical(q)
        lex = rag._lexical_scores(bigram, q)
        qv = provider.embed([q], is_query=True)[0]
        sem = [cosine(qv, dv) for dv in doc_vecs]
        def topk(sc):
            ix = [i for i, _ in sorted(enumerate(sc), key=lambda p: -p[1])[:K_PROD]]
            return [i for i in ix if sc[i] > 0]
        lt, st = topk(lex), topk(sem)
        union = list(dict.fromkeys(lt + st))
        def rrf(i):
            s = 0.0
            if i in lt: s += 1.0 / (rag._RRF_K + lt.index(i) + 1)
            if i in st: s += 1.0 / (rag._RRF_K + st.index(i) + 1)
            return s
        return sorted(union, key=rrf, reverse=True)

    configs = {"纯词法": run_lexical, "混合·生产(交集+RRF)": run_prod, "混合·并集+RRF": run_union}
    hits: dict[str, list[bool]] = {}
    for name, fn in configs.items():
        hits[name] = [fn(q)[:1] == [e] for q, e in labeled]

    print(f"provider = {provider.name if provider else 'None（纯词法）'}   有答案查询 n = {n}")
    print()
    print(f"{'配置':<22}{'命中':>5}{'Hit@1':>8}{'95%CI(正态)':>18}{'95%CI(Wilson)':>20}")
    for name, hs in hits.items():
        k = sum(hs)
        lo_n, hi_n = normal_ci(k, n)
        lo_w, hi_w = wilson_ci(k, n)
        print(f"{name:<22}{k:>5}{100*k/n:>7.1f}%"
              f"{f'  [{100*lo_n:.1f}, {100*hi_n:.1f}]':>18}"
              f"{f'  [{100*lo_w:.1f}, {100*hi_w:.1f}]':>20}")
    print()
    se = math.sqrt(0.778 * 0.222 / n)
    print(f"书中公式对照：n={n}, p≈77.8%  ->  SE = {100*se:.1f}pt,  95% CI ≈ ±{100*1.96*se:.1f}pt")
    print(f"  => 本评估集下，小于约 {100*1.96*se:.0f} 个百分点的差异无法与抽样噪声区分。")
    print()

    names = list(configs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ha, hb = hits[a], hits[b]
            bb = sum(1 for x, y in zip(ha, hb) if x and not y)   # a 对 b 错
            cc = sum(1 for x, y in zip(ha, hb) if y and not x)   # a 错 b 对
            p_mc = mcnemar_exact(bb, cc)
            delta, lo, hi, p_bs = paired_bootstrap(ha, hb)
            verdict = "显著" if p_mc < 0.05 else "不显著"
            print(f"— {a}  vs  {b}")
            print(f"    命中 {sum(ha)} vs {sum(hb)}   差值 {100*delta:+.1f}pt")
            print(f"    不一致对: 仅{a}对={bb}  仅{b}对={cc}")
            print(f"    McNemar 精确检验 p = {p_mc:.4f}  ->  {verdict}")
            print(f"    配对 bootstrap 差值 95%CI = [{100*lo:+.1f}, {100*hi:+.1f}]pt, p = {p_bs:.4f}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
