"""跨会话记忆测试。

验证「学情轨迹」这条记忆链路真的能跨会话闭环：
1. 上次写进去的学情，下次能读回来；
2. 多次会话会累积，且读到的是最近一次；
3. 记忆读写失败时不阻塞主流程（故障隔离）。

全部使用临时 SQLite 文件，不污染项目的 learning_agent.db。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from app import db
from app.agents.memory_agent import build_load_memory_node, build_save_memory_node
from app.models import Review


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """把 db 层的 engine 换到临时文件，跑完即弃。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mem_test.db'}", future=True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    db.init_db()
    return engine


def test_load_memory_reads_previous_session(tmp_db) -> None:
    """第一次会话写入的学情，第二次会话应能读回。"""
    build_save_memory_node(
        {
            "user_id": "u-001",
            "review": Review(
                mastery="入门阶段",
                strengths=["目标清晰"],
                gaps=["算法薄弱"],
                suggestions=["每天刷 1 道 LeetCode"],
            ),
            "plan": None,
            "errors": [],
        }
    )

    out = build_load_memory_node({"user_id": "u-001", "errors": []})

    assert out["memory_hits"] == 1
    assert out["memory"] is not None
    assert out["memory"].mastery == "入门阶段"
    assert "算法薄弱" in out["memory"].gaps
    assert out["errors"] == []


def test_sessions_accumulate_and_read_latest(tmp_db) -> None:
    """多次会话应累积，且 load 返回的是最近一次。"""
    for i, (mastery, gap) in enumerate(
        [("入门阶段", "算法薄弱"), ("进阶中", "工程经验不足")], start=1
    ):
        build_save_memory_node(
            {
                "user_id": "u-002",
                "review": Review(mastery=mastery, gaps=[gap]),
                "plan": None,
                "errors": [],
            }
        )
        assert db.count_sessions("u-002") == i

    out = build_load_memory_node({"user_id": "u-002", "errors": []})

    assert out["memory_hits"] == 2
    assert out["memory"].mastery == "进阶中"  # 最近一次
    assert "工程经验不足" in out["memory"].gaps


def test_memory_failure_does_not_block_pipeline(tmp_db, monkeypatch) -> None:
    """记忆层炸了也必须能继续：只记一笔 error，不抛异常、不吞掉主流程。"""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("sqlite 不可用")

    monkeypatch.setattr(db, "count_sessions", _boom)
    monkeypatch.setattr(db, "load_last_session", _boom)

    out = build_load_memory_node({"user_id": "u-003", "errors": []})

    assert out["memory"] is None
    assert out["memory_hits"] == 0
    assert any("load_memory 失败" in e for e in out["errors"])

    monkeypatch.setattr(db, "save_session", _boom)
    out2 = build_save_memory_node(
        {
            "user_id": "u-003",
            "review": Review(mastery="入门"),
            "plan": None,
            "errors": [],
        }
    )
    assert any("save_memory 失败" in e for e in out2["errors"])
