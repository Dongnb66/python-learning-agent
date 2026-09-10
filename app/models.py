"""数据模型层。

包含：
- 对话消息 Message
- 6 维学生画像 Dimension / Profile（A3 赛题核心）
- 学习计划 Plan、资源 ResourceItem、测验 Quiz、复盘 Review
- LangGraph 共享状态 AgentState
"""
from __future__ import annotations

from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 对话与画像
# --------------------------------------------------------------------------- #
class Dimension(BaseModel):
    """画像单个维度：等级 + 具体说明。"""

    level: str = Field(
        ...,
        description="水平或强度等级，如 beginner/intermediate/advanced 或 high/medium/low",
    )
    details: str = Field(default="", description="该维度的具体描述")


class Profile(BaseModel):
    """A3 赛题定义的 6 维学生画像。"""

    name: Optional[str] = None
    major: Optional[str] = None
    knowledgeBase: Dimension = Field(..., description="知识基础")
    learningGoal: Dimension = Field(..., description="学习目标")
    cognitiveStyle: Dimension = Field(..., description="认知风格")
    weakPoints: Dimension = Field(..., description="薄弱点")
    resourcePreference: Dimension = Field(..., description="资源偏好")
    studyTime: Dimension = Field(..., description="可用学习时间")
    summary: str = Field(default="", description="画像一句话总结")


class Message(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str


# --------------------------------------------------------------------------- #
# 计划 / 资源 / 测验 / 复盘
# --------------------------------------------------------------------------- #
class PlanStep(BaseModel):
    step: int
    title: str
    description: str
    est_minutes: int


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)
    total_minutes: int = 0


class ResourceItem(BaseModel):
    title: str
    url: Optional[str] = None
    type: str = Field(default="article", description="article/video/course/doc")
    description: str = ""
    relevance: str = ""


class ResourceList(BaseModel):
    """资源列表的包装模型。

    function calling 需要 Pydantic 类来生成工具 schema，裸 list[...] 泛型不行，
    故包一层。
    """

    items: List[ResourceItem] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    q: str
    options: List[str] = Field(default_factory=list)
    answer: str = ""
    explanation: str = ""


class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(default_factory=list)


class Review(BaseModel):
    mastery: str = ""
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    comparison: str = Field(
        default="",
        description="与上一次学习会话的纵向对比（新学员首轮为空字符串）",
    )


class LearningSession(BaseModel):
    """一次学习会话的学情快照——跨会话记忆的持久化单元。

    每次走完完整管线，ReviewAgent 的复盘结论会被存成一条 session。
    下次同一 user_id 再来时，load_memory 节点把它读出来喂给复盘智能体，
    从而产出「上次 vs 本次」的纵向对比。
    """

    id: Optional[int] = None
    user_id: str = ""
    created_at: str = ""
    goal: str = ""
    mastery: str = ""
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 自主辅导智能体（ReAct）产物
# --------------------------------------------------------------------------- #
class ToolCallRecord(BaseModel):
    """一轮工具调用的审计记录 —— 自主决策轨迹的可复核证据。"""

    round: int = Field(default=0, description="第几轮循环（从 1 开始）")
    tool: str = Field(default="", description="被调用的工具名")
    args: str = Field(default="", description="工具入参（JSON 文本）")
    observation: str = Field(default="", description="工具返回的观察结果摘要")


class TutorSession(BaseModel):
    """自主辅导智能体一次运行的完整结果。

    trace 是「自主决策」的可审计证据：面试/复现时可直接看到
    模型在第几轮、调了哪个工具、观察到了什么、何时决定停止。
    """

    question: str = Field(default="", description="触发本次辅导的诊断问题")
    answer: str = Field(default="", description="综合工具观察后给出的辅导结论")
    tools_used: List[str] = Field(default_factory=list, description="本轮实际调用过的工具名（去重保序）")
    rounds: int = Field(default=0, description="实际执行的 ReAct 循环轮数")
    mode: str = Field(default="llm", description="llm = 真实模型驱动；policy = 无 Key 时的规则策略兜底")
    trace: List[ToolCallRecord] = Field(default_factory=list, description="逐轮工具调用轨迹")


# --------------------------------------------------------------------------- #
# LangGraph 共享状态
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    user_id: str
    messages: List[Message]
    profile: Profile
    plan: Optional[Plan]
    resources: List[ResourceItem]
    quiz: Optional[Quiz]
    review: Optional[Review]
    # ---- 跨会话记忆 ----
    memory: Optional[LearningSession]  # 上一次会话的学情（load_memory 载入）
    memory_hits: int  # 该用户历史会话总数，0 表示新学员
    errors: List[str]
    # ---- 自主辅导智能体（ReAct）----
    tutoring: Optional[TutorSession]  # 自主工具调用循环的产物
    agent_trace: List[ToolCallRecord]  # 自主决策轨迹（可审计）


# --------------------------------------------------------------------------- #
# API 请求 / 响应
# --------------------------------------------------------------------------- #
class ProfileBuildRequest(BaseModel):
    user_id: str
    messages: List[Message]


class LearnRequest(BaseModel):
    user_id: str
    messages: List[Message]


class LearnResponse(BaseModel):
    user_id: str
    profile: Profile
    plan: Optional[Plan] = None
    resources: List[ResourceItem] = Field(default_factory=list)
    quiz: Optional[Quiz] = None
    review: Optional[Review] = None
    tutoring: Optional[TutorSession] = Field(
        default=None, description="自主辅导智能体（ReAct）的结论与决策轨迹"
    )
    errors: List[str] = Field(default_factory=list)
