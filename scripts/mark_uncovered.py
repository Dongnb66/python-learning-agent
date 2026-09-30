# -*- coding: utf-8 -*-
"""给「语料未覆盖」的查询打标记 —— 不改分组、不删数据，只标注并报两个口径。

背景
----
人工复核 3 条可疑标注时发现：它们的问题不是「标注指错了文档」，而是
**期望文档里根本没有能回答提问的内容**。语料每篇只有 76–100 字符（入门级导语），
而查询在问具体细节：

  | 查询 | 期望文档 | 文档里有没有 |
  |---|---|---|
  | ORM 怎么建模型和写查询 | SQLAlchemy 2.0 | ORM ✗ / 模型 ✗ / 查询 ✗（只写 DeclarativeBase、Session） |
  | 请求头和状态码分别是什么意思 | 计算机网络 | 请求头 ✗ / 状态码 ✗（只写三次握手、DNS） |
  | 刷题应该从哪一类题开始做 | 数据结构与算法 | 刷题 ✗ / 从哪 ✗（只写「技术面试必刷内容」） |

这三条**天然不可能被检索命中**（实测全部 miss），却计在 36 条有答案查询的分母里。

处理原则
--------
**不删、不改指向、不改分组** —— 只在数据里打 `_语料未覆盖` 标记，
让基准脚本能同时报「含它们」与「不含它们」两个口径。

理由：删掉数据等于把不利证据藏起来；改指向等于编一个「正确答案」。
**标注出来 + 报两个口径，才是可核查的做法。**
"""
from __future__ import annotations

import io
import json
import os

QFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "eval", "retrieval_queries.json")

# 查询原文 -> 该文档缺哪些关键词（人工复核时确认的）
UNCOVERED = {
    "ORM 怎么建模型和写查询": {
        "缺失关键词": ["ORM", "对象关系", "模型", "查询"],
        "文档实际讲的": "DeclarativeBase、Mapped 类型注解、Session 上下文管理",
        "判定": "标注指的文档主题是对的（SQLAlchemy 就是 ORM），但文档通篇没用「ORM/模型/查询」这些词",
    },
    "请求头和状态码分别是什么意思": {
        "缺失关键词": ["请求头", "状态码", "Header"],
        "文档实际讲的": "TCP 三次握手、HTTP 与 HTTPS、DNS 解析",
        "判定": "标注指的文档主题是对的（HTTP 概念），但文档没讲请求头与状态码",
    },
    "刷题应该从哪一类题开始做": {
        "缺失关键词": ["刷题", "从哪", "开始"],
        "文档实际讲的": "覆盖数组/链表/哈希表/二叉树，「技术面试必刷内容」「建议每天 1-2 题」",
        "判定": "语义路把它排第 1（模型认得出「刷题」→LeetCode），但文档没回答「从哪一类开始」",
    },
}


def main() -> int:
    data = json.load(io.open(QFILE, encoding="utf-8"))
    n_marked = 0
    for grp in ("A", "B"):
        for item in data.get(grp, []):
            q = item.get("q", "")
            if q in UNCOVERED:
                item["_语料未覆盖"] = UNCOVERED[q]
                n_marked += 1
    if "_口径" in data:
        data["_口径"] += (
            f" 其中 {n_marked} 条带 `_语料未覆盖` 标记 —— 查询问的内容在期望文档里不存在，"
            "天然不可能命中。报告应同时给出「含它们」与「不含它们」两个口径，不得只报其一。"
        )
    io.open(QFILE, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"已标记 {n_marked} 条语料未覆盖查询")
    for grp in ("A", "B"):
        for item in data.get(grp, []):
            if "_语料未覆盖" in item:
                print(f"  [{grp}] {item['q']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
