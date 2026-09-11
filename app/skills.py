"""技能层 —— Skill 懒加载（元数据常驻上下文，正文按需读取）。

解决什么问题
------------
自主 Agent 的领域知识如果整篇塞进系统提示词，上下文会被「可能用不上」的
长文本占满：技能越多浪费越大，且大部分轮次根本用不到。

本模块把每个技能拆成两段，按不同成本供给模型：

    ┌──────────────────────────────┬───────────────────────────┐
    │ 元数据（名称/说明/适用时机）     │ 正文（完整操作步骤）        │
    │ 常驻系统提示词，每条几十 token   │ 只在命中场景时经工具读一次  │
    │ 读取成本 = 只扫 frontmatter     │ 用完即弃，不长期占用上下文  │
    └──────────────────────────────┴───────────────────────────┘

即「目录永远在手边，正文按需翻」——技能数量增长时，常驻上下文只按
元数据条数线性增长，与正文长度无关。

三个设计点（面试可讲）
----------------------
1. **懒加载是真的懒**：list_skills 只扫描每个文件开头的 frontmatter
   （读到第二个 `---` 即停，超过 _FRONTMATTER_MAX_BYTES 视为格式错误跳过），
   不读取正文；正文只经 load_skill 按需进入上下文。
2. **防幻觉同源**：load_skill 收到未知技能名时返回可用清单并明确拒绝，
   绝不编造「大概有这个技能」——与 BM25 空命中拒答是同一条设计原则。
3. **零依赖可单测**：frontmatter 手写解析（不引 yaml），目录可参数注入，
   纯文件操作不碰 DB / LLM。

文件格式（app/data/skills/*.md）
--------------------------------
---
name: skill-name
description: 一句话说明（会出现在常驻技能清单里）
when_to_use: 什么场景下应该加载本技能
---
（以下为正文：完整操作指南）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 技能文件所在目录（正文与元数据同文件，用 frontmatter 分隔）
SKILLS_DIR = Path(__file__).resolve().parent / "data" / "skills"

# frontmatter 扫描上限：防止「忘了写闭合 ---」的文件把整篇正文当元数据读进来
_FRONTMATTER_MAX_BYTES = 4096

# 元数据必填字段；缺任一字段的技能文件视为格式错误，跳过并记录
_REQUIRED_FIELDS = ("name", "description", "when_to_use")


@dataclass(frozen=True)
class SkillMeta:
    """技能元数据——这是唯一「常驻」进入模型上下文的部分。"""

    name: str
    description: str
    when_to_use: str
    body_chars: int  # 正文长度，用于在清单里提示「读它会花多少上下文」
    file: Path  # 源文件路径，供 load_skill 精确定位（正文按需读取用）


def _read_frontmatter(path: Path) -> tuple[dict[str, str], int]:
    """只读文件开头，解析 frontmatter，返回 (字段字典, 正文起始字节偏移)。

    懒加载的关键：读到 frontmatter 闭合的 `---` 就停止解析；
    正文字节数用文件总大小减去偏移得到，同样不需要读入正文。
    按**原始字节**切行再解析（\r 仅在取值时剥掉），偏移量对 LF / CRLF 都精确。
    """
    with path.open("rb") as fh:
        head = fh.read(_FRONTMATTER_MAX_BYTES)
    raw_lines = head.split(b"\n")

    if not raw_lines or raw_lines[0].rstrip(b"\r") != b"---":
        return {}, 0
    fields: dict[str, str] = {}
    for idx, raw in enumerate(raw_lines[1:], start=1):
        line = raw.rstrip(b"\r")
        if line == b"---":
            offset = sum(len(ln) + 1 for ln in raw_lines[: idx + 1])
            return fields, offset
        key_b, sep, val_b = line.partition(b":")
        if sep:
            fields[key_b.strip().decode("utf-8")] = val_b.strip().decode("utf-8")
    return {}, 0  # 没找到闭合 ---：格式错误


def list_skills(skills_dir: Path | str | None = None) -> list[SkillMeta]:
    """扫描技能目录，返回全部技能的元数据（不读取任何正文内容）。"""
    root = Path(skills_dir) if skills_dir else SKILLS_DIR
    if not root.is_dir():
        return []

    metas: list[SkillMeta] = []
    for path in sorted(root.glob("*.md")):
        size = path.stat().st_size
        fields, offset = _read_frontmatter(path)
        if any(not fields.get(f) for f in _REQUIRED_FIELDS):
            continue  # frontmatter 缺字段：视为无效技能，跳过
        metas.append(
            SkillMeta(
                name=fields["name"],
                description=fields["description"],
                when_to_use=fields["when_to_use"],
                body_chars=max(size - offset, 0),
                file=path,
            )
        )
    return metas


def find_skill(name: str, skills_dir: Path | str | None = None) -> SkillMeta | None:
    """按名查找技能元数据；不存在返回 None（调用方据此拒答，不许编造）。"""
    lowered = (name or "").strip().lower()
    return next((m for m in list_skills(skills_dir) if m.name.lower() == lowered), None)


def load_skill_body(name: str, skills_dir: Path | str | None = None) -> str | None:
    """按需读取指定技能的完整正文；技能不存在时返回 None。"""
    meta = find_skill(name, skills_dir)
    if meta is None:
        return None
    # 精确跳过 frontmatter 偏移，只把正文读进上下文
    with meta.file.open("rb") as fh:
        fh.seek(_offset_of(meta.file))
        return fh.read().decode("utf-8", errors="replace").strip()


def _offset_of(path: Path) -> int:
    """重新计算某技能文件的正文偏移（与 _read_frontmatter 保持同一实现口径）。"""
    fields, offset = _read_frontmatter(path)
    return offset if fields else 0


def available_names(skills_dir: Path | str | None = None) -> str:
    """渲染「可用技能名」清单文本，供拒答信息与提示词共用。"""
    return "、".join(m.name for m in list_skills(skills_dir)) or "（暂无可用技能）"


def skills_metadata_prompt(skills_dir: Path | str | None = None) -> str:
    """渲染常驻系统提示词的「技能清单」段落。

    这里只出现元数据（名称/说明/适用时机/正文长度），
    完整正文必须经 load_skill 工具按需读取——这是懒加载的入口约定。
    """
    metas = list_skills(skills_dir)
    if not metas:
        return ""
    lines = [
        "",
        "## 可用技能（懒加载：清单常驻，正文按需读取）",
        "以下是你可以使用的技能目录。需要某个技能的**完整操作步骤**时，"
        "调用 load_skill(skill_name) 读取正文；不要凭清单自行编造具体步骤。",
        "",
    ]
    for m in metas:
        lines.append(
            f"- **{m.name}**（正文约 {m.body_chars} 字符）：{m.description}"
            f"｜适用：{m.when_to_use}"
        )
    lines.append("")
    return "\n".join(lines)
