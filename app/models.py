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
    errors: List[str]


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
    errors: List[str] = Field(default_factory=list)
