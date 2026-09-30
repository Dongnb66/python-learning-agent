# -*- coding: utf-8 -*-
"""给 60 条标注集加上「语料指纹」，并把评测口径拆开。

为什么需要语料指纹
------------------
`eval/retrieval_queries.json` 里的 expected 是**语料下标**。语料一改（增删改任一
篇、甚至只是调整标题），下标就会错位、答案就不再成立 —— 但**基准脚本不会报错**，
它只会安静地算出一堆错数字。这是「评测集与被测索引强耦合」问题：三份 RAG 失效
模式文档都只讲索引版本，没人讲**基准确也带版本**。

做法：把语料的内容指纹（篇数 + 每篇 url/正文的 sha256）写进查询集，
基准脚本每次运行前核对；不一致就**显式失败**，而不是继续算。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

QFILE = os.path.join(_ROOT, "eval", "retrieval_queries.json")


def corpus_fingerprint() -> dict:
    """语料指纹：篇数 + 逐篇 (url, sha256(标题+正文)) 的汇总哈希。

    用汇总哈希而不是逐篇列表，是为了让查询集里那一行短到能一眼看完；
    不一致时再由基准脚本打出逐篇差异。
    """
    from app import rag

    docs = rag._load_documents()
    per = []
    for d in docs:
        h = hashlib.sha256((d.metadata.get("title", "") + "\n" + d.page_content).encode("utf-8")).hexdigest()
        per.append((d.metadata.get("url", ""), h))
    per.sort()
    joined = "\n".join(f"{u}\t{h}" for u, h in per)
    return {
        "篇数": len(docs),
        "汇总sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16],
        "逐篇": [{"url": u, "sha256": h[:16]} for u, h in per],
    }


def main() -> int:
    fp = corpus_fingerprint()
    data = json.load(io.open(QFILE, encoding="utf-8"))
    old = data.get("_语料指纹")

    data["_语料指纹"] = fp
    # 口径说明也写进数据文件，避免「口径」只活在脚本注释里
    data["_口径"] = (
        "A+B = 36 条有标注答案（用于 Hit@k / MRR）；C = 22 条语料外（用于正确拒答率）；"
        "C_争议 = 2 条单列，不并入 C 主口径。"
        "⚠️ Hit@k 只在 36 条有答案的查询上计算，拒答率只在 22 条语料外上计算 —— "
        "两者是不同口径，不得合成一个数对外报。"
    )

    with io.open(QFILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"语料 {fp['篇数']} 篇，指纹 {fp['汇总sha256']}")
    if old:
        print(f"  旧指纹 {old.get('汇总sha256')}  篇数 {old.get('篇数')}  "
              + ("（未变）" if old.get("汇总sha256") == fp["汇总sha256"] else "⚠️ 变了！核对 expected 下标是否仍成立"))
    else:
        print("  首次写入指纹")
    print(f"已更新 {QFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
