"""持久化层：用 SQLAlchemy 2.0 把学生画像存入 SQLite。

仅做轻量持久化，便于演示「学情可追溯」。生产可换成 Postgres。
"""
from __future__ import annotations

from sqlalchemy import Engine, create_engine, desc, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import get_settings
from app.models import Dimension, LearningSession, Profile


class Base(DeclarativeBase):
    pass


class ProfileRecord(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str | None] = mapped_column(default=None)
    major: Mapped[str | None] = mapped_column(default=None)
    knowledge_base: Mapped[str] = mapped_column(default="")
    learning_goal: Mapped[str] = mapped_column(default="")
    cognitive_style: Mapped[str] = mapped_column(default="")
    weak_points: Mapped[str] = mapped_column(default="")
    resource_preference: Mapped[str] = mapped_column(default="")
    study_time: Mapped[str] = mapped_column(default="")
    summary: Mapped[str] = mapped_column(default="")
    raw_json: Mapped[str] = mapped_column(default="")

    def to_profile(self) -> Profile:
        def _dim(raw: str) -> Dimension:
            if not raw:
                return Dimension(level="unknown", details="")
            try:
                import json

                d = json.loads(raw)
                return Dimension(level=d.get("level", "unknown"), details=d.get("details", ""))
            except Exception:
                return Dimension(level="unknown", details=raw)

        return Profile(
            name=self.name,
            major=self.major,
            knowledgeBase=_dim(self.knowledge_base),
            learningGoal=_dim(self.learning_goal),
            cognitiveStyle=_dim(self.cognitive_style),
            weakPoints=_dim(self.weak_points),
            resourcePreference=_dim(self.resource_preference),
            studyTime=_dim(self.study_time),
            summary=self.summary,
        )


class SessionRecord(Base):
    """学情轨迹表：每条记录是一次完整学习会话的复盘结论。

    与 profiles 表（长期画像）配合，构成跨会话记忆的两层持久化：
    - profiles        → 长期画像（这个学生是谁）
    - learning_sessions → 学情轨迹（这个学生历次学得怎么样）
    """

    __tablename__ = "learning_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(index=True)
    created_at: Mapped[str] = mapped_column(default="")
    goal: Mapped[str] = mapped_column(default="")
    mastery: Mapped[str] = mapped_column(default="")
    strengths: Mapped[str] = mapped_column(default="")
    gaps: Mapped[str] = mapped_column(default="")
    suggestions: Mapped[str] = mapped_column(default="")
    raw_json: Mapped[str] = mapped_column(default="")

    def to_session(self) -> LearningSession:
        import json

        def _lst(raw: str) -> list[str]:
            if not raw:
                return []
            try:
                v = json.loads(raw)
                return v if isinstance(v, list) else [str(v)]
            except Exception:
                return [raw]

        return LearningSession(
            id=self.id,
            user_id=self.user_id,
            created_at=self.created_at,
            goal=self.goal,
            mastery=self.mastery,
            strengths=_lst(self.strengths),
            gaps=_lst(self.gaps),
            suggestions=_lst(self.suggestions),
        )


def get_engine() -> Engine:
    s = get_settings()
    return create_engine(s.database_url, future=True)


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


def save_profile(user_id: str, profile: Profile) -> None:
    import json

    engine = get_engine()
    with Session(engine) as session:
        rec = session.scalar(select(ProfileRecord).where(ProfileRecord.user_id == user_id))
        if rec is None:
            rec = ProfileRecord(user_id=user_id)
        rec.name = profile.name
        rec.major = profile.major
        rec.knowledge_base = json.dumps(profile.knowledgeBase.model_dump(), ensure_ascii=False)
        rec.learning_goal = json.dumps(profile.learningGoal.model_dump(), ensure_ascii=False)
        rec.cognitive_style = json.dumps(profile.cognitiveStyle.model_dump(), ensure_ascii=False)
        rec.weak_points = json.dumps(profile.weakPoints.model_dump(), ensure_ascii=False)
        rec.resource_preference = json.dumps(
            profile.resourcePreference.model_dump(), ensure_ascii=False
        )
        rec.study_time = json.dumps(profile.studyTime.model_dump(), ensure_ascii=False)
        rec.summary = profile.summary
        rec.raw_json = profile.model_dump_json(ensure_ascii=False)
        session.add(rec)
        session.commit()


def load_profile(user_id: str) -> Profile | None:
    engine = get_engine()
    with Session(engine) as session:
        rec = session.scalar(select(ProfileRecord).where(ProfileRecord.user_id == user_id))
        return rec.to_profile() if rec else None


# --------------------------------------------------------------------------- #
# 学情轨迹（跨会话记忆的第二层）
# --------------------------------------------------------------------------- #
def save_session(user_id: str, review, goal: str = "") -> int:
    """把一次会话的复盘结论写入学情轨迹表，返回新记录 id。"""
    import json
    from datetime import datetime, timezone

    engine = get_engine()
    with Session(engine) as session:
        rec = SessionRecord(
            user_id=user_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            goal=goal or "",
            mastery=getattr(review, "mastery", "") or "",
            strengths=json.dumps(
                list(getattr(review, "strengths", []) or []), ensure_ascii=False
            ),
            gaps=json.dumps(list(getattr(review, "gaps", []) or []), ensure_ascii=False),
            suggestions=json.dumps(
                list(getattr(review, "suggestions", []) or []), ensure_ascii=False
            ),
            raw_json=(
                review.model_dump_json(ensure_ascii=False)
                if hasattr(review, "model_dump_json")
                else ""
            ),
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec.id


def load_last_session(user_id: str) -> LearningSession | None:
    """读取该用户最近一次学情快照；没有历史则返回 None（新学员）。"""
    engine = get_engine()
    with Session(engine) as session:
        rec = session.scalar(
            select(SessionRecord)
            .where(SessionRecord.user_id == user_id)
            .order_by(desc(SessionRecord.id))
            .limit(1)
        )
        return rec.to_session() if rec else None


def count_sessions(user_id: str) -> int:
    """统计该用户历史会话数（0 = 新学员）。"""
    engine = get_engine()
    with Session(engine) as session:
        return (
            session.scalar(
                select(func.count())
                .select_from(SessionRecord)
                .where(SessionRecord.user_id == user_id)
            )
            or 0
        )
