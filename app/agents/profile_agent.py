"""ProfileAgent —— 学生 6 维画像抽取（A3 赛题核心）。

对应原 Node 版 profileAgent.js：
- 用 LLM 结构化输出抽取 6 维画像
- 用正则额外提取「姓名」「专业」，作为高置信信号覆盖 LLM 结果
- 生成一句话画像总结
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import dialogue_text, to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, Dimension, Profile

# 姓名提取：覆盖「我叫/我是/我的名字是/姓名是」等常见表述
_NAME_PATTERNS = [
    re.compile(r"(?:我叫|我是|我的名字是|姓名是|大家好?，?我?叫)\s*([^\s,，。.!！?？]{1,10})"),
]
_NAME_BLACKLIST = {"小明", "小红", "老师", "同学", "大家", "我们", "自己", "那个人"}

# 专业提取：XX专业 / XX系 / XX学院 / XX方向。
# 先吃掉「我是/我在/我读/他/是/在/的」等填充词，再抓 2~8 字专业名，
# 并用否定/疑问词黑名单过滤「没有任何专业」「不记得是什么专业」这类非专业表述。
_MAJOR_PATTERN = re.compile(
    r"(?:我是|我在|我读|我学|他是|她是|你是|就读|学习|专业是|我|他|她|你|是|在|的|于)*"
    r"([^\s,，。.!！?？]{2,8})(?:专业|系|学院|方向)"
)
_MAJOR_BAD_WORDS = ("没有", "任何", "什么", "哪", "啥", "不", "无", "没")


def extract_name(text: str) -> str | None:
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip()
            if cand and cand not in _NAME_BLACKLIST:
                return cand
    return None


def extract_major(text: str) -> str | None:
    m = _MAJOR_PATTERN.search(text)
    if not m:
        return None
    cand = m.group(1).strip()
    if not cand or any(bad in cand for bad in _MAJOR_BAD_WORDS):
        return None
    return cand


_PROFILE_SYSTEM = """你是一名「学生画像分析智能体」。
任务：根据师生对话，抽取学生的 6 维学习画像，每维给出 level 和 details。
六维定义：
1. knowledgeBase 知识基础（level: beginner/intermediate/advanced）
2. learningGoal 学习目标（level: vague/clear/ambitious，details 写具体目标）
3. cognitiveStyle 认知风格（如 视觉型/逻辑型/动手型，details 写依据）
4. weakPoints 薄弱点（level: low/medium/high 表示薄弱程度，details 写具体短板）
5. resourcePreference 资源偏好（如 视频/图文/项目驱动，details 写偏好原因）
6. studyTime 可用学习时间（level: low/medium/high，details 写每周大致小时数）
只输出符合 Profile schema 的 JSON，不要解释。"""


def build_profile_node(state: AgentState) -> dict:
    chat = get_structured_model(Profile, temperature=0.2)

    messages = (
        [SystemMessage(content=_PROFILE_SYSTEM)]
        + to_lc_messages(state["messages"])
        + [HumanMessage(content="请基于以上对话输出学生画像 JSON。")]
    )
    profile: Profile = chat.invoke(messages)

    # 高置信正则覆盖——只扫学生（user）消息。
    # 若把 assistant 消息也算进来，「你好！我是学习画像助手」这类
    # 系统开场白会被误提取成学生姓名/专业。
    user_text = dialogue_text([m for m in state["messages"] if m.role == "user"])
    name = extract_name(user_text)
    if name:
        profile.name = name
    major = extract_major(user_text)
    if major:
        profile.major = major

    # 生成一句话总结（若 LLM 未给）
    if not profile.summary:
        profile.summary = _summarize(profile)

    return {"profile": profile}


def _summarize(p: Profile) -> str:
    parts = []
    if p.name:
        parts.append(p.name)
    if p.major:
        parts.append(f"{p.major}专业")
    level = p.knowledgeBase.level
    goal = p.learningGoal.details or p.learningGoal.level
    parts.append(f"基础{level}")
    if goal:
        parts.append(f"目标：{goal}")
    return "，".join(parts) + "。" if parts else "画像已生成。"
