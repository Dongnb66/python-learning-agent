"""持久化层：用 SQLAlchemy 2.0 把学生画像存入 SQLite。

仅做轻量持久化，便于演示「学情可追溯」。生产可换成 Postgres。
"""
from __future__ import annotations

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import get_settings
from app.models import Dimension, Profile


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
