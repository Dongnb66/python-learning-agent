"""HuggingFace Spaces / Render 在线 Demo —— 个性化学习多智能体系统。

运行模式（自动判断）：
- 设置了环境变量 LLM_API_KEY  →  真实模式，调用 DeepSeek（需在 Space Secrets 配置）。
- 未设置 LLM_API_KEY          →  Mock 模式（默认），无需任何 Key 即可展示完整 5 智能体流程。

本项目即 github.com/Dongnb66/python-learning-agent：基于 LangGraph 编排的
「画像 → 计划 → 资源 → 测验 → 复盘」多智能体学习系统，资源推荐走 RAG 防幻觉。
"""
from __future__ import annotations

import os

import gradio as gr

import mock_llm

# 没有 API Key 就走 mock，保证 Spaces 零配置可跑
USE_MOCK = not bool(os.getenv("LLM_API_KEY"))
if USE_MOCK:
    mock_llm.install()

from app.graph import build_graph  # noqa: E402
from app.models import AgentState, Message  # noqa: E402

graph = build_graph()

PLACEHOLDER = """叫我杨运栋，软件工程专业。
Python 基础一般，算法比较薄弱。
我想做 AI Agent 方向，每周能学 10 小时，偏好视频和动手项目。"""


def _parse_dialogue(text: str) -> list[Message]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return [Message(role="user", content=ln) for ln in lines]


def _fmt_profile(p):
    if p is None:
        return "_（画像为空）_"
    rows = []
    for label, dim in [
        ("知识基础", p.knowledgeBase),
        ("学习目标", p.learningGoal),
        ("认知风格", p.cognitiveStyle),
        ("薄弱点", p.weakPoints),
        ("资源偏好", p.resourcePreference),
        ("学习时间", p.studyTime),
    ]:
        rows.append(f"| {label} | {dim.level} | {dim.details} |")
    head = f"**姓名**：{p.name or '（未提供）'}  ｜  **专业**：{p.major or '（未提供）'}\n\n"
    if p.summary:
        head += f"> {p.summary}\n\n"
    head += "| 维度 | 等级 | 说明 |\n|---|---|---|\n" + "\n".join(rows)
    return head


def _fmt_plan(plan):
    if plan is None:
        return "_（计划为空）_"
    rows = [f"| {s.step} | {s.title} | {s.description} | {s.est_minutes} 分钟 |" for s in plan.steps]
    return (
        f"**总目标**：{plan.goal}\n\n"
        f"| 步骤 | 标题 | 说明 | 预估耗时 |\n|---|---|---|---|\n" + "\n".join(rows)
        + f"\n\n**累计耗时**：{plan.total_minutes} 分钟"
    )


def _fmt_resources(resources):
    if not resources:
        return "_（无推荐资源）_"
    rows = [f"- [{r.title}]({r.url or '#'}) ｜ 类型：{r.type} ｜ {r.relevance}" for r in resources]
    return "**以下资源全部来自本地资料库真实链接（RAG 防幻觉）：**\n\n" + "\n".join(rows)


def _fmt_quiz(quiz):
    if quiz is None or not quiz.questions:
        return "_（无测验）_"
    blocks = []
    for i, q in enumerate(quiz.questions, 1):
        opts = "\n".join(f"  - {o}" for o in q.options)
        blocks.append(f"**{i}. {q.q}**\n{opts}\n> 答案：{q.answer} ｜ {q.explanation}")
    return "\n\n".join(blocks)


def _fmt_review(rv):
    if rv is None:
        return "_（无复盘）_"
    return (
        f"**掌握度**：{rv.mastery}\n\n"
        f"**优势**：{', '.join(rv.strengths) or '—'}\n\n"
        f"**差距**：{', '.join(rv.gaps) or '—'}\n\n"
        f"**建议**：\n" + "\n".join(f"- {s}" for s in rv.suggestions)
    )


def run(dialogue: str):
    if not dialogue.strip():
        return ("请输入你的学习情况（多行均可）",) * 5
    messages = _parse_dialogue(dialogue)
    state: AgentState = {
        "user_id": "demo-user",
        "messages": messages,
        "resources": [],
        "errors": [],
    }
    final = graph.invoke(state)
    return (
        _fmt_profile(final.get("profile")),
        _fmt_plan(final.get("plan")),
        _fmt_resources(final.get("resources", [])),
        _fmt_quiz(final.get("quiz")),
        _fmt_review(final.get("review")),
    )


mode_text = "🟢 Mock 模式（无需 API Key，展示完整流程）" if USE_MOCK else "🔵 真实模式（DeepSeek）"

with gr.Blocks(title="个性化学习多智能体系统 · Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 个性化学习多智能体系统 · 在线 Demo\n"
        "基于 **LangGraph** 编排的 5 智能体工作流：`画像 → 计划 → 资源(RAG防幻觉) → 测验 → 复盘`。\n\n"
        f"当前模式：**{mode_text}**  ｜  源码：[github.com/Dongnb66/python-learning-agent](https://github.com/Dongnb66/python-learning-agent)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="用自然语言告诉我你的学习情况（每行一句）",
                placeholder=PLACEHOLDER,
                value=PLACEHOLDER,
                lines=8,
            )
            btn = gr.Button("🚀 运行多智能体流程", variant="primary")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("① 学习画像"):
                    out_profile = gr.Markdown()
                with gr.TabItem("② 学习计划"):
                    out_plan = gr.Markdown()
                with gr.TabItem("③ 资源推荐(RAG)"):
                    out_resources = gr.Markdown()
                with gr.TabItem("④ 自测题"):
                    out_quiz = gr.Markdown()
                with gr.TabItem("⑤ 学情复盘"):
                    out_review = gr.Markdown()
    btn.click(
        run,
        inputs=inp,
        outputs=[out_profile, out_plan, out_resources, out_quiz, out_review],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
