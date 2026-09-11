"""Skill 懒加载（元数据常驻 / 正文按需读）的回归测试。

覆盖四个关键性质：
1. list_skills 只产出元数据，正文内容不会混进常驻清单；
2. load_skill 按需返回完整正文；
3. 未知技能名返回 None / 拒答信息，绝不编造（与 BM25 空命中拒答同源）；
4. tutor 的工具注册表与 LLM 路径的常驻提示词都包含技能清单。
"""
from __future__ import annotations

import pytest

from app import skills as skills_registry
from app.agents import tutor_agent
from app.tools import TOOL_REGISTRY

# --------------------------------------------------------------------------- #
# 造假技能目录：正文远大于元数据，用来验证「清单不掺正文」
# --------------------------------------------------------------------------- #
_HUGE_BODY = "\n".join(f"这是很长的正文第 {i} 行，用于验证懒加载。" for i in range(200))


@pytest.fixture
def skill_dir(tmp_path):
    (tmp_path / "alpha.md").write_text(
        "---\nname: alpha\ndescription: 测试技能 A\nwhen_to_use: 需要技能 A 时\n---\n"
        + _HUGE_BODY,
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text(
        "---\nname: beta\ndescription: 测试技能 B\nwhen_to_use: 需要技能 B 时\n---\n正文 B",
        encoding="utf-8",
    )
    # 格式错误：缺 when_to_use → 必须被跳过，不许半残进入清单
    (tmp_path / "broken.md").write_text(
        "---\nname: broken\ndescription: 缺字段的技能\n---\n正文", encoding="utf-8"
    )
    return tmp_path


def test_list_skills_returns_metadata_and_skips_invalid(skill_dir) -> None:
    metas = skills_registry.list_skills(skill_dir)
    names = [m.name for m in metas]
    assert names == ["alpha", "beta"]  # broken 被跳过，且按文件名有序
    for m in metas:
        assert m.description and m.when_to_use
        assert m.body_chars > 0


def test_metadata_prompt_never_contains_body(skill_dir) -> None:
    """懒加载的核心断言：常驻清单里只有元数据，正文一行都不许进。"""
    prompt = skills_registry.skills_metadata_prompt(skill_dir)
    assert "alpha" in prompt and "beta" in prompt
    # 取正文里独有的句子，断言它没被整段带进常驻上下文
    assert "这是很长的正文第 199 行" not in prompt
    assert "正文 B" not in prompt


def test_load_skill_returns_full_body(skill_dir) -> None:
    body = skills_registry.load_skill_body("alpha", skill_dir)
    assert body is not None and len(body) > 1000  # 完整正文，不被 frontmatter 截断
    assert "这是很长的正文第 199 行" in body
    assert skills_registry.load_skill_body("beta", skill_dir) == "正文 B"


def test_unknown_skill_returns_none_and_tool_refuses(tmp_path) -> None:
    """防幻觉：未知技能名 → 元数据层返回 None，工具层返回拒答 + 可用清单，不编造。"""
    assert skills_registry.load_skill_body("no-such-skill", tmp_path) is None
    assert skills_registry.find_skill("no-such-skill", tmp_path) is None

    obs = TOOL_REGISTRY["load_skill"].invoke({"skill_name": "no-such-skill"})
    assert "没有名为" in obs
    assert "可用技能" in obs


def test_real_skill_pack_loaded_and_registered() -> None:
    """真实技能包：至少 3 个技能、全部进了工具注册表、且正文各不相同。"""
    metas = skills_registry.list_skills()
    assert len(metas) >= 3
    assert "load_skill" in TOOL_REGISTRY

    bodies = [skills_registry.load_skill_body(m.name) for m in metas]
    assert all(bodies), "每个技能的正文都必须可读且非空"
    assert len(set(bodies)) == len(bodies), "技能正文不允许互相重复"


def test_tutor_system_prompt_contains_skill_catalog() -> None:
    """LLM 路径的常驻提示词必须带技能清单（懒加载入口），且正文不进提示词。"""
    prompt = tutor_agent._SYSTEM + skills_registry.skills_metadata_prompt()
    assert "load_skill" in prompt
    assert "可用技能" in prompt
    for meta in skills_registry.list_skills():
        assert meta.name in prompt
