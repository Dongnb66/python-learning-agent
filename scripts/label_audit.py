# -*- coding: utf-8 -*-
"""标注集机械校验 —— 用数据筛出「最可能是错标」的查询，并生成人工抽检表。

为什么做这个
------------
`eval/retrieval_queries.json` 的 60 条是 **AI 标注**（弱监督），而它是整条评测链的地基：
简历头条的 19.4% → 77.8%、min_score=2.0 的选型、消融结论、C 组拒答率 —— 全部建在它上面。

**《深入理解 AI Agent》第 7 章里，基准质量的唯一来源都是人工**：
SWE-bench Verified 从 2294 个任务随机抽 1699 个、招 93 名精通 Python 的开发者逐条检查，
最终仅 500 个通过（淘汰率 71%）。

人工复核只能由人做，但**机械校验能先筛一遍**：如果某条查询的「期望文档」在
词法分与语义分里都排得很靠后，那这条标注就很可疑 —— 人工只需要看这些。

⚠️ 这**不能替代**人工复核。它只减少要看多少条。

用法：
    EMBED_BACKEND=local-onnx EMBED_LOCAL_MODEL_DIR=<目录> python scripts/label_audit.py
    （不配 embedding 也能跑，只是少了语义路那一列）
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import rag  # noqa: E402
from app.embeddings import cosine, get_embedding_provider  # noqa: E402

OUT = os.path.join(_ROOT, "eval", "label_audit_20260930.md")


def main() -> int:
    corpus = rag._load_documents()
    titles = [d.metadata.get("title", "")[:38] for d in corpus]
    qs = json.load(open(os.path.join(_ROOT, "eval", "retrieval_queries.json"), encoding="utf-8"))

    provider = get_embedding_provider()
    doc_vecs = provider.embed([d.page_content for d in corpus]) if provider else None
    bigram = rag._get_bigram_retriever(k=4)

    def rank_of(scores, idx):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order.index(idx) + 1 if idx in order else 99

    rows = []
    for grp in ("A", "B"):
        for item in qs[grp]:
            q, exp = item["q"], item.get("expected")
            lex = rag._lexical_scores(bigram, q)
            lr = rank_of(lex, exp)
            if doc_vecs is not None:
                qv = provider.embed([q], is_query=True)[0]
                sem = [cosine(qv, dv) for dv in doc_vecs]
                sr = rank_of(sem, exp)
            else:
                sr = None
            worst = max(lr, sr) if sr else lr
            rows.append({"组": grp, "q": q, "expected": exp, "词法排名": lr,
                         "语义排名": sr, "期望文档": titles[exp], "worst": worst})

    rows.sort(key=lambda r: -r["worst"])

    lines = [
        "# 标注集机械校验与人工抽检表",
        "",
        f"- 语料：{len(corpus)} 篇　provider：{provider.name if provider else '（未配，只有词法列）'}",
        f"- 待核：A 组 {len(qs['A'])} 条 + B 组 {len(qs['B'])} 条 = {len(rows)} 条有标注答案",
        "",
        "## 判读方式",
        "",
        "「排名」= 该查询的**期望文档**在全部 12 篇里的名次（1 = 最相关）。",
        "**词法与语义两路都排在后面**（worst ≥ 7，即掉出上半区）的，最可能是错标。",
        "",
        "⚠️ 但**排名低不等于标错**：B 组本来就是「词面不匹配」的同义改写，",
        "若语义路也认不出，可能是**这条查询确实超出了语料范围**（该归到 C 组）。",
        "这需要你来判断 —— 我能给的只是「哪几条最该看」。",
        "",
        "## 全部 36 条（按可疑度降序）",
        "",
        "| 可疑度 | 组 | 查询 | 期望文档 | 词法排名 | 语义排名 | 你的判断 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sr = r["语义排名"] if r["语义排名"] is not None else "-"
        mark = "🔴" if r["worst"] >= 9 else ("🟡" if r["worst"] >= 7 else "🟢")
        lines.append(f"| {mark} {r['worst']} | {r['组']} | {r['q']} | {r['期望文档']} | "
                     f"{r['词法排名']} | {sr} | |")
    lines += [
        "",
        "## 怎么填",
        "",
        "「你的判断」列三选一：",
        "- `对` —— 期望文档确实是这条查询该命中的",
        "- `错` —— 该命中别的文档（请在后面注明是第几篇）",
        "- `超范围` —— 这条其实语料里没有答案，应归到 C 组",
        "",
        "填完统计 `对 / (对+错+超范围)` 就是 **AI-人工一致率**。",
        "这个数字比 60 条标注本身更值得写进简历：**「自建 60 条标注集，人工抽检 N 条、一致率 X%」**。",
        "",
        "## 建议的抽检量",
        "",
        "- **最小版**：只看 🔴（可疑度 ≥ 9）那几条",
        "- **标准版**：🟡 及以上（可疑度 ≥ 7），约 10 条以内",
        "- **完整版**：全部 36 条",
        "",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    red = [r for r in rows if r["worst"] >= 9]
    yellow = [r for r in rows if 7 <= r["worst"] < 9]
    print(f"待核 {len(rows)} 条")
    print(f"  🔴 可疑度 ≥9（强烈建议人工看）：{len(red)} 条")
    print(f"  🟡 可疑度 7-8（标准版含）：{len(yellow)} 条")
    print(f"  🟢 其余 {len(rows) - len(red) - len(yellow)} 条")
    print()
    print("  最可疑的 8 条：")
    for r in rows[:8]:
        sr = r["语义排名"] if r["语义排名"] is not None else "-"
        print(f"    [{r['组']}] 词法{r['词法排名']:>2} 语义{sr:>2}  {r['q'][:26]}")
        print(f"          期望 → {r['期望文档']}")
    print()
    print(f"已写出人工抽检表：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
