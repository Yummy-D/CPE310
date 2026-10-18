"""ORM models for StoolCheck AI.

PRIVACY: only non-sensitive metadata is persisted.
No image paths, no image bytes, no health content beyond the Bristol type.
Phase 5: user accounts + analyses linked to users via a real Foreign Key.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def _utcnow() -> datetime:
    """Timezone-naive UTC timestamp (SQLite stores no tz info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    """A registered account (Phase 5).

    email  = unique login identifier (used for authentication)
    username = unique display name shown in the UI
    password is stored bcrypt-hashed.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)


class Analysis(Base):
    """One stool-image analysis result, owned by a user."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), index=True, nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow
    )
    bristol_type: Mapped[int] = mapped_column(Integer, nullable=False)
