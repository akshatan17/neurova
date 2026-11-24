"""
The database: four tables.

  users     people who can sign in (patient, clinician or admin)
  access    which clinician a patient has chosen to share their results with
  studies   one analysed scan or sleep recording, with the model's result
  audit     a log of who did what, which is never edited or deleted

SQLite is used by default (a single file, nothing to install). To use MySQL
instead, set DATABASE_URL, for example
    mysql+pymysql://user:password@localhost/neurova
"""
import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, create_engine, inspect, text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATA_DIR = os.environ.get("NEUROVA_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
URL = os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(DATA_DIR, "neurova.db"))

engine = create_engine(URL, connect_args={"check_same_thread": False} if URL.startswith("sqlite") else {})
Session = sessionmaker(engine, expire_on_commit=False)
BigText = Text().with_variant(LONGTEXT, "mysql")      # results include images, which outgrow MySQL's TEXT


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(190), unique=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(12))                 # patient | clinician | admin
    created: Mapped[datetime] = mapped_column(DateTime, default=now)
    active: Mapped[bool] = mapped_column(Boolean, default=True)   # an admin can disable an account


class Access(Base):
    """A row here means: this patient lets this clinician see their results."""
    __tablename__ = "access"
    __table_args__ = (UniqueConstraint("patient_id", "clinician_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    clinician_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    granted: Mapped[datetime] = mapped_column(DateTime, default=now)


class Study(Base):
    __tablename__ = "studies"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(8))                  # mri | sleep
    title: Mapped[str] = mapped_column(String(200))
    headline: Mapped[str] = mapped_column(String(120))            # e.g. "Glioma, 97%"
    needs_review: Mapped[bool] = mapped_column(Boolean, default=True)
    result_json: Mapped[str] = mapped_column(BigText)             # everything the model returned
    created: Mapped[datetime] = mapped_column(DateTime, default=now)
    # filled in when a clinician reviews the result
    review_verdict: Mapped[str | None] = mapped_column(String(10), nullable=True)   # agree | disagree
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Audit(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    user_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(12))
    action: Mapped[str] = mapped_column(String(40))
    detail: Mapped[str] = mapped_column(String(300), default="")


def log(db, user, action: str, detail: str = ""):
    """Add a line to the audit log. `user` may be None for a failed sign-in."""
    db.add(Audit(user_name=user.name if user else "unknown", role=user.role if user else "-", action=action, detail=detail[:300]))


Base.metadata.create_all(engine)

# A database made by an earlier version has no "active" column yet: add it, so nobody has to start over.
if "active" not in {c["name"] for c in inspect(engine).get_columns("users")}:
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE users ADD COLUMN active BOOLEAN NOT NULL DEFAULT 1"))
