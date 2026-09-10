"""A3 兼容层的数据存储。

A3 前端除了画像/学习流程外，还会产生这些持久化数据：
- 资源包（/api/generate 的产物，资源详情页按 id 读取）
- 前后测记录（/api/efficacy/submit）
- 知识库文档（/api/knowledge 的增改查）
- 自定义模型（/api/models 的增删）
- 系统日志（/api/analytics 的统计源）

这些数据结构与 python-learning-agent 原有的 SQLAlchemy 表不同，
为了不动现有 64 条测试所依赖的表结构，这里用独立 JSON 文件存储：
零迁移、零依赖，读写均在进程内加锁，满足 Demo 与测试场景。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "a3_store.json")

_lock = threading.RLock()


def _store_path() -> str:
    """存储文件路径，支持用 A3_STORE_PATH 环境变量重定向（测试用）。"""
    return os.getenv("A3_STORE_PATH") or _DEFAULT_PATH


def _empty() -> dict:
    return {"resources": [], "efficacy": [], "knowledge": [], "customModels": [], "logs": []}


def load() -> dict:
    with _lock:
        path = _store_path()
        if not os.path.exists(path):
            return _empty()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return _empty()
        merged = _empty()
        for k in merged:
            if isinstance(data.get(k), list):
                merged[k] = data[k]
        return merged


def save(store: dict) -> None:
    with _lock:
        path = _store_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)


def _mutate(fn) -> dict:
    """读-改-写一次完成，避免并发下互相覆盖。fn 接收并返回 store dict。"""
    with _lock:
        store = load()
        out = fn(store)
        save(out)
        return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 资源包
# --------------------------------------------------------------------------- #
def add_resource(pkg: dict) -> None:
    _mutate(lambda s: {**s, "resources": [pkg, *s["resources"]][:200]})


def get_resource(rid: str) -> dict | None:
    return next((r for r in load()["resources"] if r.get("id") == rid), None)


def list_resources() -> list[dict]:
    return load()["resources"]


# --------------------------------------------------------------------------- #
# 前后测记录
# --------------------------------------------------------------------------- #
def add_efficacy(record: dict) -> None:
    _mutate(lambda s: {**s, "efficacy": [record, *s["efficacy"]][:1000]})


def list_efficacy() -> list[dict]:
    return load()["efficacy"]


# --------------------------------------------------------------------------- #
# 知识库文档
# --------------------------------------------------------------------------- #
def list_knowledge() -> list[dict]:
    return load()["knowledge"]


def get_knowledge(doc_id: str) -> dict | None:
    return next((d for d in load()["knowledge"] if d.get("id") == doc_id), None)


def upsert_knowledge(doc: dict) -> dict:
    def _fn(s: dict) -> dict:
        docs = [d for d in s["knowledge"] if d.get("id") != doc["id"]]
        docs.insert(0, doc)
        return {**s, "knowledge": docs}

    _mutate(_fn)
    return doc


def delete_knowledge(doc_id: str) -> bool:
    def _fn(s: dict) -> dict:
        docs = [d for d in s["knowledge"] if d.get("id") != doc_id]
        return {**s, "knowledge": docs}

    _mutate(_fn)
    return get_knowledge(doc_id) is None


# --------------------------------------------------------------------------- #
# 自定义模型
# --------------------------------------------------------------------------- #
def list_custom_models() -> list[dict]:
    return load()["customModels"]


def upsert_custom_model(model: dict) -> None:
    def _fn(s: dict) -> dict:
        models = [m for m in s["customModels"] if m.get("id") != model["id"]]
        models.insert(0, model)
        return {**s, "customModels": models}

    _mutate(_fn)


def delete_custom_model(model_id: str) -> bool:
    def _fn(s: dict) -> dict:
        models = [m for m in s["customModels"] if m.get("id") != model_id]
        return {**s, "customModels": models}

    _mutate(_fn)
    return not any(m["id"] == model_id for m in load()["customModels"])


# --------------------------------------------------------------------------- #
# 系统日志
# --------------------------------------------------------------------------- #
def add_log(log_type: str, latency_ms: int = 0) -> None:
    def _fn(s: dict) -> dict:
        logs = [{"type": log_type, "latency": latency_ms, "createdAt": _now()}, *s["logs"]][:1000]
        return {**s, "logs": logs}

    _mutate(_fn)
