"""Mock LLM 层 —— 让 Demo / 评测在没有任何 API Key 的情况下也能跑通完整多智能体流程。

原理：
- 真实项目里每个 Agent 都通过 `app.llm.get_structured_model(Schema)` 拿到一个
  `.invoke(messages) -> Schema 实例` 的对象；
- 本模块提供一个「假模型」，根据传入的 Schema 类型返回一份**写死的、结构合法**的样例输出；
- 触发方式二选一：`MOCK_LLM=1` 环境变量（`app.llm` 内检测），
  或在导入 app 模块前显式调用 `install()`。

为什么放在 app/ 下：
- `app.llm` 需要按需导入它，放在包内可避免「从仓库根目录导模块」的隐式依赖；
- 仓库根目录的 `mock_llm.py` 保留为兼容 shim，供 `app.py`（Gradio Demo）沿用旧路径。

关键设计（保真度）：
- ProfileAgent 的「姓名/专业正则提取」是真实代码，会基于用户输入覆盖 mock 结果，
  所以 Demo 里填的名字/专业是真的会显示的；
- ResourceAgent 的 BM25 检索是真实代码，本模块返回的资源**直接取自
  `app/data/resources.json`**（标题/URL/类型全部来自真实语料），
  因此「只推荐真实链接」走的是真链路，不是写死的假数据。
"""
from __future__ import annotations

import json
import os
from typing import Any

from app.models import (
    Dimension,
    Plan,
    PlanStep,
    Profile,
    Quiz,
    QuizQuestion,
    ResourceItem,
    ResourceList,
    Review,
)

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "resources.json")

# 这些语料标题会被 mock 选中作为推荐结果（全部真实存在于资料库）
_PICK_TITLES = [
    "FastAPI 官方教程",
    "LangGraph 多智能体编排入门",
    "RAG 检索增强生成原理与实践",
    "Docker 容器化 Python 应用",
]


def _load_real_resources() -> list[dict]:
    try:
        with open(_CORPUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


_REAL = _load_real_resources()


def _sample_resources() -> ResourceList:
    """从真实资料库里取若干条——保证 mock 输出与防幻觉白名单天然一致。"""
    by_title = {r.get("title"): r for r in _REAL if r.get("title")}
    chosen = [by_title[t] for t in _PICK_TITLES if t in by_title] or list(_REAL[:4])

    items: list[ResourceItem] = []
    for meta in chosen:
        items.append(
            ResourceItem(
                title=meta.get("title", ""),
                url=meta.get("url") or None,
                type=meta.get("type", "doc"),
                description=(meta.get("content", "") or "")[:60] + "…",
                relevance="与学习计划中的对应步骤直接匹配，且为官方 / 权威资料。",
            )
        )
    return ResourceList(items=items)


# --------------------------------------------------------------------------- #
# 各 Agent 的样例输出
# --------------------------------------------------------------------------- #
def _sample_profile() -> Profile:
    return Profile(
        name=None,  # 由 ProfileAgent 正则从用户输入提取并覆盖
        major=None,
        knowledgeBase=Dimension(level="intermediate", details="Python 基础一般，能写简单脚本"),
        learningGoal=Dimension(level="clear", details="想做 AI Agent 方向，目标是能独立交付一个 Agent 项目"),
        cognitiveStyle=Dimension(level="逻辑型", details="偏好先理解原理再动手"),
        weakPoints=Dimension(level="high", details="算法与数据结构薄弱"),
        resourcePreference=Dimension(level="视频", details="偏好视频 + 动手项目"),
        studyTime=Dimension(level="medium", details="每周约 10 小时"),
        summary="",
    )


def _sample_plan() -> Plan:
    steps = [
        PlanStep(step=1, title="Python 与 FastAPI 基础巩固", description="补齐类型注解、Pydantic、异步视图，能独立写一个小 API。", est_minutes=180),
        PlanStep(step=2, title="LangChain / LangGraph 多智能体入门", description="理解 StateGraph 如何把多个 Agent 串成有状态工作流。", est_minutes=240),
        PlanStep(step=3, title="RAG 检索增强实践", description="用 BM25 接入本地资料库，理解防幻觉机制。", est_minutes=200),
        PlanStep(step=4, title="动手做一个校园场景 Agent", description="把前几步串起来，做一个能跑的最小可用 Agent。", est_minutes=300),
        PlanStep(step=5, title="部署与工程化（Docker）", description="容器化并部署到 HuggingFace / Render，形成作品集。", est_minutes=150),
    ]
    return Plan(goal="从零构建并部署第一个 AI Agent 项目", steps=steps, total_minutes=sum(s.est_minutes for s in steps))


def _sample_quiz() -> Quiz:
    return Quiz(
        questions=[
            QuizQuestion(
                q="LangGraph 中用什么把多个智能体串成有状态工作流？",
                options=["StateGraph", "普通函数调用", "SQL 触发器", "CSS 动画"],
                answer="StateGraph",
                explanation="StateGraph 用节点+边定义多智能体流程，支持状态持久化与条件分支。",
            ),
            QuizQuestion(
                q="RAG 主要通过什么手段缓解大模型幻觉？",
                options=["把相关资料检索后拼进提示", "增大模型参数量", "更换 GPU", "减少训练数据"],
                answer="把相关资料检索后拼进提示",
                explanation="RAG 用检索到的真实资料作为事实依据约束生成，降低编造风险。",
            ),
            QuizQuestion(
                q="FastAPI 依靠哪个库实现请求体类型校验与自动文档？",
                options=["Pydantic", "NumPy", "Pandas", "Requests"],
                answer="Pydantic",
                explanation="FastAPI 基于 Pydantic 做数据校验，并据此自动生成 OpenAPI 文档。",
            ),
        ]
    )


def _sample_review() -> Review:
    return Review(
        mastery="基础入门、方向明确，已具备构建第一个 Agent 的知识框架",
        strengths=["目标清晰，方向聚焦 AI Agent", "已具备 Python 基础", "学习意愿强、每周时间充足"],
        gaps=["算法与数据结构需补齐", "工程化部署经验不足"],
        suggestions=["每天 1-2 道 LeetCode 打底", "先完整跑通本项目再扩展功能", "把项目部署到 HuggingFace 形成可点开的简历作品集"],
    )


_SAMPLES = {
    "Profile": _sample_profile,
    "Plan": _sample_plan,
    "ResourceList": _sample_resources,
    "Quiz": _sample_quiz,
    "Review": _sample_review,
}


class _FakeStructured:
    """替代 get_structured_model 返回的对象：invoke() 直接给出写死的样例。"""

    def __init__(self, schema):
        self._schema = schema

    def invoke(self, messages=None):  # noqa: ANN001
        factory = _SAMPLES.get(self._schema.__name__)
        if factory is None:
            raise RuntimeError(f"mock 未覆盖 schema: {self._schema.__name__}")
        return factory()


def get_mock_structured_model(schema: Any) -> _FakeStructured:
    """给 `app.llm.get_structured_model` 用的假模型工厂。"""
    return _FakeStructured(schema)


def get_mock_chat_model() -> None:
    """mock 模式下不应走到真实 ChatModel 调用路径，直接给出可诊断的错误。

    例外：TutorAgent 的 ReAct 循环在没有可用 Key 时**不会**调用本函数
    ——它会走规则策略（policy），同样完成「决策 → 调工具 → 观察」循环，
    因此零 Key 环境下仍可验证自主决策的循环结构。
    """
    raise RuntimeError("mock 模式下不应调用真实 ChatModel；请检查调用路径。")


def install() -> None:
    """猴子补丁：把 `app.llm` 的两个工厂函数换成假模型。

    注意：若 Agent 模块已先被导入（它们是 `from app.llm import ...` 的
    名字绑定），打补丁对已绑定的引用无效。因此推荐直接用
    `MOCK_LLM=1` 环境变量——`app.llm` 在**函数调用时**判断，不受导入顺序影响。
    """
    import app.llm as llm_mod

    llm_mod.get_structured_model = lambda schema, **_: get_mock_structured_model(schema)  # type: ignore[assignment]
    llm_mod.get_chat_model = lambda **_: get_mock_chat_model()  # type: ignore[assignment]
