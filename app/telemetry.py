"""运行时可观测层 —— 让「这条管线花了多少、错在哪、防幻觉拦了几次」变成可查的数字。

为什么需要它
------------
本项目此前的「可验证」只覆盖到**离线**：全量单测 + 评测集 + 消融实验，
都是提交前跑一次、把结论写进报告。
但真实服务跑起来之后还有一类问题没人回答得了：

- 这次请求走了哪些节点？哪个节点最慢？慢在模型调用还是检索？
- 自主辅导 Agent 真的在调工具吗？调了几次？还是每次都在空转？
- 防幻觉的三道约束**到底有没有被触发过**？如果没有，是真的没有幻觉，
  还是约束根本没生效（写错了、被短路了）？

前两条靠 trace 回答，第三条靠**把守卫的拦截动作变成计数器**回答 ——
「我做了防幻觉」是声明，「线上拒答 N 次、拦下编造链接 M 条」是证据。

三层结构
--------
    Trace（一次请求一条）        ── 含若干 Span，可查「哪个节点慢」
      └─ Span（一个节点/一次模型调用）
    Registry（进程级累计）        ── 计数器 + 各节点的耗时/次数/错误数，供 /api/metrics
    Guard 计数器                 ── 拒答 / 拦下编造链接 / 工具调用 …… 防幻觉的运行时证据

设计取舍
--------
1. **零依赖、纯标准库**：不引 opentelemetry。理由是本项目要能在零 Key、无网的
   环境里跑通与单测；引入 SDK 只会让「离线可复现」这条卖点变脆。字段命名
   刻意向 OTel / Prometheus 的习惯靠（span / duration_ms / `<name>_total`），
   将来真接 OpenTelemetry 只需换一层导出器，不改埋点。
2. **埋点位置在编排层，不在每个 Agent 内部**：节点级 span 由 graph.py 构建图时
   统一包装，Agent 代码保持干净 —— 加监控不该污染业务逻辑。
3. **contextvar 而非全局变量传递**：LangGraph 同步 `invoke` 与 FastAPI 的
   线程池都在同一线程内向下调用，contextvar 能天然做到「一次请求一条 trace」、
   并发请求互不串台，且不需要把 trace 塞进 AgentState 污染状态结构。
4. **有界存储**：trace 只保留最近 `_TRACE_KEEP` 条。内存里存全量历史等于埋一个
   缓慢泄漏；要长期留存应由外部系统（日志/时序库）承担。
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

# --------------------------------------------------------------------------- #
# 计数器词表（集中定义 = 单一事实来源，避免各处自由发挥写错名字）
# --------------------------------------------------------------------------- #
C_REQUESTS = "requests_total"  # 收到的业务请求数
C_LLM_CALLS = "llm_calls_total"  # 模型调用次数（真实 + 模拟）
C_LLM_MOCK_CALLS = "llm_mock_calls_total"  # 其中属于 Mock 模型的部分（真实调用 = 前者 - 后者）
C_LLM_ERRORS = "llm_errors_total"  # 模型调用抛错次数
C_TOOL_CALLS = "tool_calls_total"  # 自主 Agent 的工具调用次数
C_RETRIEVAL = "retrieval_calls_total"  # 检索次数
C_RETRIEVAL_EMPTY = "retrieval_empty_total"  # 检索被阈值过滤后为空的次数（防幻觉第①道）
C_REFUSALS = "refusals_total"  # 因无事实依据而代码级拒答的次数（防幻觉第②道）
C_FABRICATED_BLOCKED = "fabricated_links_blocked_total"  # 被白名单拦下的编造链接条数（第③道）
C_RESOURCES_RETURNED = "resources_returned_total"  # 最终返回给用户的资源条数

_TRACE_KEEP = 50  # 内存中保留的最近 trace 条数
_MOCK_TRUTHY = {"1", "true", "yes", "on"}


def mock_mode() -> bool:
    """当前是否处于 Mock 模式（转发 app.llm 的判断，避免两处各写一套真值规则）。"""
    try:
        from app.llm import mock_enabled  # 延迟导入：避免 telemetry ↔ llm 循环导入

        return bool(mock_enabled())
    except Exception:  # pragma: no cover - 仅在极端导入环境下兜底
        return (os.getenv("MOCK_LLM") or "").strip().lower() in _MOCK_TRUTHY


# --------------------------------------------------------------------------- #
# Span / Trace
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    """一次可计时的执行单元（一个图节点、一次模型调用、一次检索）。"""

    name: str
    started_at: float
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error
    error: str = ""
    parent: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }
        if self.parent:
            d["parent"] = self.parent
        if self.error:
            d["error"] = self.error
        if self.attrs:
            d["attrs"] = self.attrs
        return d


@dataclass
class Trace:
    """一次请求的完整链路。"""

    trace_id: str
    started_at: float
    spans: list[Span] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "ok"
    error: str = ""

    def to_dict(self, *, with_spans: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "span_count": len(self.spans),
            "counters": dict(self.counters),
        }
        if self.error:
            d["error"] = self.error
        if with_spans:
            d["spans"] = [s.to_dict() for s in self.spans]
        return d

    def slowest(self, n: int = 3) -> list[dict[str, Any]]:
        """耗时最长的 n 个 span —— 「哪个节点慢」的答案。"""
        top = sorted(self.spans, key=lambda s: s.duration_ms, reverse=True)[:n]
        return [{"name": s.name, "duration_ms": s.duration_ms} for s in top]

    def node_spans(self) -> list[Span]:
        return [s for s in self.spans if s.name.startswith("node.")]


# --------------------------------------------------------------------------- #
# 进程级指标注册表
# --------------------------------------------------------------------------- #
class _Registry:
    """进程级累计指标。线程安全（FastAPI 的同步端点跑在线程池里）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._span_count: dict[str, int] = {}
        self._span_ms: dict[str, float] = {}
        self._span_errors: dict[str, int] = {}
        self._started = time.time()

    def bump(self, name: str, n: float = 1) -> None:
        if n == 0:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def observe(self, name: str, duration_ms: float, *, error: bool = False) -> None:
        with self._lock:
            self._span_count[name] = self._span_count.get(name, 0) + 1
            self._span_ms[name] = self._span_ms.get(name, 0.0) + duration_ms
            if error:
                self._span_errors[name] = self._span_errors.get(name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            names = sorted(self._span_count)
            spans = {
                nm: {
                    "count": self._span_count[nm],
                    "total_ms": round(self._span_ms.get(nm, 0.0), 2),
                    "avg_ms": round(self._span_ms.get(nm, 0.0) / self._span_count[nm], 2),
                    "errors": self._span_errors.get(nm, 0),
                }
                for nm in names
            }
            uptime = time.time() - self._started

        return {
            "mock_llm": mock_mode(),
            "uptime_seconds": round(uptime, 1),
            "counters": counters,
            "spans": spans,
            # 模型调用：真实 / 模拟分开，避免把离线压测的数字当成 API 调用量
            "llm_calls": {
                "total": counters.get(C_LLM_CALLS, 0),
                "mock": counters.get(C_LLM_MOCK_CALLS, 0),
                "real": counters.get(C_LLM_CALLS, 0) - counters.get(C_LLM_MOCK_CALLS, 0),
                "errors": counters.get(C_LLM_ERRORS, 0),
            },
            # 防幻觉的运行时证据：把三道约束的拦截动作单独拎出来，便于一眼核对
            "guardrails": {
                "retrieval_empty_total": counters.get(C_RETRIEVAL_EMPTY, 0),
                "refusals_total": counters.get(C_REFUSALS, 0),
                "fabricated_links_blocked_total": counters.get(C_FABRICATED_BLOCKED, 0),
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._span_count.clear()
            self._span_ms.clear()
            self._span_errors.clear()
            self._started = time.time()


REGISTRY = _Registry()

# trace 有界存储：只留最近 N 条
_TRACE_STORE: "OrderedDict[str, Trace]" = OrderedDict()
_TRACE_LOCK = threading.Lock()


def _sanitize(name: str) -> str:
    """Prometheus 指标名只允许 [a-zA-Z0-9_:]，把点/横线统一换成下划线。"""
    return "".join(ch if (ch.isalnum() or ch in "_") else "_" for ch in name)


# --------------------------------------------------------------------------- #
# 埋点原语
# --------------------------------------------------------------------------- #
_current: ContextVar[Trace | None] = ContextVar("telemetry_current_trace", default=None)
_stack: ContextVar[tuple[str, ...]] = ContextVar("telemetry_span_stack", default=())


def bump(name: str, n: float = 1) -> None:
    """累加一个计数器（进程级 + 当前 trace 各记一份）。"""
    REGISTRY.bump(name, n)
    tr = _current.get()
    if tr is not None:
        tr.counters[name] = tr.counters.get(name, 0) + n


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span | None]:
    """给一段代码计时并记录；异常会标记为 error 后原样抛出。

    没有活动 trace 时也照样计时（只进进程级指标）——这样单测可以直接用。
    """
    started = time.perf_counter()
    stack = _stack.get()
    sp = Span(name=name, started_at=time.time(), parent=stack[-1] if stack else "", attrs=dict(attrs))
    token = _stack.set(stack + (name,))
    try:
        yield sp
    except BaseException as exc:
        sp.status = "error"
        sp.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _stack.reset(token)
        sp.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        REGISTRY.observe(name, sp.duration_ms, error=sp.status == "error")
        tr = _current.get()
        if tr is not None:
            tr.spans.append(sp)


@contextlib.contextmanager
def trace_run(trace_id: str | None = None) -> Iterator[Trace]:
    """开一条新 trace，把块内所有 span / 计数器都归到它名下。"""
    tr = Trace(trace_id=trace_id or uuid.uuid4().hex[:12], started_at=time.time())
    tok_current = _current.set(tr)
    tok_stack = _stack.set(())
    try:
        yield tr
    except BaseException as exc:
        tr.status = "error"
        tr.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        tr.duration_ms = round((time.time() - tr.started_at) * 1000, 3)
        _current.reset(tok_current)
        _stack.reset(tok_stack)
        _store(tr)


def _store(tr: Trace) -> None:
    with _TRACE_LOCK:
        _TRACE_STORE[tr.trace_id] = tr
        while len(_TRACE_STORE) > _TRACE_KEEP:
            _TRACE_STORE.popitem(last=False)


# --------------------------------------------------------------------------- #
# 读取接口（供 /api/metrics、/api/traces）
# --------------------------------------------------------------------------- #
def snapshot() -> dict[str, Any]:
    """进程级指标快照（JSON 友好）。"""
    return REGISTRY.snapshot()


def latest_trace_ids(n: int = 10) -> list[str]:
    with _TRACE_LOCK:
        return list(_TRACE_STORE)[-n:][::-1]


def get_trace(trace_id: str) -> dict[str, Any] | None:
    with _TRACE_LOCK:
        tr = _TRACE_STORE.get(trace_id)
    return tr.to_dict() if tr else None


def reset() -> None:
    """清空全部指标与 trace —— 供单测隔离用。"""
    REGISTRY.reset()
    with _TRACE_LOCK:
        _TRACE_STORE.clear()


def render_prometheus() -> str:
    """把指标渲染成 Prometheus 文本格式（`GET /api/metrics?format=prometheus`）。

    不引 prometheus_client：本项目的定位是「零 Key 可复现」，能直接用
    `curl`/`txt2prom` 抓走即可；真接 Prometheus 也只差一个 exporter。
    """
    snap = REGISTRY.snapshot()
    lines: list[str] = []

    lines.append("# HELP learning_agent_mock_llm 1 表示当前跑在 Mock 模型(非真实API)下")
    lines.append("# TYPE learning_agent_mock_llm gauge")
    lines.append(f"learning_agent_mock_llm {1 if snap['mock_llm'] else 0}")

    lines.append("# HELP learning_agent_uptime_seconds 进程已运行秒数")
    lines.append("# TYPE learning_agent_uptime_seconds gauge")
    lines.append(f"learning_agent_uptime_seconds {snap['uptime_seconds']}")

    lines.append("# HELP learning_agent_counter 业务计数器")
    lines.append("# TYPE learning_agent_counter counter")
    for name in sorted(snap["counters"]):
        lines.append(f'learning_agent_counter{{name="{name}"}} {snap["counters"][name]}')

    lines.append("# HELP learning_agent_span_duration_ms 各执行单元累计耗时")
    lines.append("# TYPE learning_agent_span_duration_ms summary")
    for name in sorted(snap["spans"]):
        s = snap["spans"][name]
        safe = _sanitize(name)
        lines.append(f'learning_agent_span_duration_ms_sum{{span="{safe}"}} {s["total_ms"]}')
        lines.append(f'learning_agent_span_duration_ms_count{{span="{safe}"}} {s["count"]}')
        if s["errors"]:
            lines.append(f'learning_agent_span_errors_total{{span="{safe}"}} {s["errors"]}')

    return "\n".join(lines) + "\n"
