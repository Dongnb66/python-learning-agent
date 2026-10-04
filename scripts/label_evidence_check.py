#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标注集机械证据校验（客观规则，可复现；不替代人的判断）

对 eval/retrieval_queries.json 里每条「应命中」的查询，在它期望文档
（app/data/resources.json 的对应下标）中找出词面最接近的那一句，作为
「这条查询在语料里到底有没有依据」的可读证据。

判据（纯机械，任何人都能复跑出同样结果）：
  score = 查询与某一句共有的「中文双字词 + 英文数字词(长度>=3)」个数
  ratio = score / 查询词数
  score < 2  -> 记为「语料无依据候选」，建议优先人工确认（可能应归入 C 组）

注意：本脚本只做筛选与取证，不修改任何标注，也不改动任何指标口径。
      重合度低不等于标错：B 组本就是「词面不匹配」的同义改写。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "eval" / "retrieval_queries.json"
CORPUS = ROOT / "app" / "data" / "resources.json"
OUT = ROOT / "eval" / "label_evidence_20261004.md"

STOP = {"怎么", "什么", "如何", "哪些", "为什么", "可以", "一个", "这个", "那个"}


def flatten(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(flatten(x) for x in obj)
    if isinstance(obj, dict):
        return " ".join(flatten(v) for v in obj.values())
    return ""


def tokens(text: str):
    t = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5]+", " ", text.lower())
    out = []
    for w in t.split():
        if not w:
            continue
        if re.fullmatch(r"[0-9a-z]+", w):
            if len(w) >= 3:
                out.append(w)
        else:
            for i in range(len(w) - 1):
                bg = w[i:i + 2]
                if bg not in STOP:
                    out.append(bg)
    return out


def best_sentence(query: str, doc_text: str):
    qs = set(tokens(query))
    best, best_score = "", 0
    for s in re.split(r"[。！？；\n]", doc_text):
        s = s.strip()
        if len(s) <= 3:
            continue
        st = set(tokens(s))
        hit = len(qs & st)
        if hit > best_score:
            best_score, best = hit, s
    ratio = round(best_score / len(qs) * 100) if qs else 0
    return best_score, best, ratio


def main() -> int:
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    corpus_raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    docs = corpus_raw if isinstance(corpus_raw, list) else next(v for v in corpus_raw.values() if isinstance(v, list))

    rows = []
    for group in ("A", "B", "C_争议"):
        for item in queries.get(group, []):
            q = item.get("q") or item.get("query") or ""
            exp = item.get("expected")
            if exp is None or (isinstance(exp, int) and exp < 0):
                continue
            if exp >= len(docs):
                rows.append((group, q, "下标 " + str(exp) + " 越界", "", 0, 0))
                continue
            doc = docs[exp]
            title = doc.get("title") or doc.get("name") or doc.get("id") or ("资源 " + str(exp))
            score, sentence, ratio = best_sentence(q, flatten(doc))
            rows.append((group, q, title, sentence, score, ratio))

    candidates = [r for r in rows if r[4] < 2]
    lines = [
        "# 标注集机械证据校验报告（2026-10-04）",
        "",
        "> 由 scripts/label_evidence_check.py 自动生成；规则客观、可复现，不替代人工判断。",
        "> 用途：把「要人工看 36 条」缩小成「只看下面这些候选」，并为每条给出可读证据句。",
        "",
        "- 参与校验：" + str(len(rows)) + " 条（A/B 组应命中 + 争议条）",
        "- 语料：" + str(len(docs)) + " 篇（app/data/resources.json）",
        "- 判据：score = 查询与某句共有的中文双字词/英文词个数；score < 2 记为语料无依据候选",
        "- 候选：**" + str(len(candidates)) + " 条**（建议优先人工确认；若确认无依据应归入 C 组）",
        "",
        "## 一、语料无依据候选（优先看这些）",
        "",
        "| 组 | 查询 | 期望文档 | 最佳证据句 | score |",
        "|---|---|---|---|---|",
    ]
    for g, q, title, s, score, ratio in candidates:
        sent = (s[:60] + "…") if len(s) > 60 else (s or "（该文档里没有任何一句与查询有共同词）")
        lines.append("| " + g + " | " + q + " | " + title + " | " + sent + " | " + str(score) + " |")
    lines += [
        "",
        "## 二、全部查询的证据句与重合度",
        "",
        "| 组 | 查询 | 期望文档 | 最佳证据句 | score | ratio |",
        "|---|---|---|---|---|---|",
    ]
    for g, q, title, s, score, ratio in sorted(rows, key=lambda r: r[4]):
        sent = (s[:70] + "…") if len(s) > 70 else (s or "（无）")
        lines.append("| " + g + " | " + q + " | " + title + " | " + sent + " | " + str(score) + " | " + str(ratio) + "% |")
    lines += [
        "",
        "## 三、边界说明（必须一起读）",
        "",
        "1. 本报告是机械筛选，不是标注结论：重合度低不等于标错。",
        "2. B 组本来就是「词面不匹配」的同义改写 —— 它的低重合是设计使然，不能据此判错。",
        "3. 真正要人工确认的是第一节那批候选：它们连一句共享词都找不出来。",
        "4. 本脚本不修改任何标注、不改动任何指标口径（主口径仍是 36 条全算）。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("候选 " + str(len(candidates)) + " / 参与 " + str(len(rows)) + " 条 -> " + str(OUT.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
