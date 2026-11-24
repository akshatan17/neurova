"""
Signing in, and deciding who may see what.

Passwords are stored only as bcrypt hashes. After signing in, the browser holds
a signed token (a JWT) and sends it with every request; the token says who the
user is and expires after twelve hours.
"""
import hashlib
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException

from .database import DATA_DIR, Access, Session, User

TOKEN_HOURS = 12


def _secret() -> str:
    """The key that signs tokens. Made once and kept in data/, unless NEUROVA_SECRET is set."""
    if os.environ.get("NEUROVA_SECRET"):
        return os.environ["NEUROVA_SECRET"]
    path = os.path.join(DATA_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    return open(path).read().strip()


SECRET = _secret()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def password_stamp(user: User) -> str:
    """A short fingerprint of the stored password hash. It goes into the token, so when the
    password is reset every token issued before stops working."""
    return hashlib.sha256(user.password_hash.encode()).hexdigest()[:12]


def make_token(user: User) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": str(user.id), "pw": password_stamp(user), "exp": expires}, SECRET, algorithm="HS256")


# ------------------------------------------------------------------ too many wrong passwords
_failures: dict[str, list[float]] = {}


def too_many_attempts(email: str) -> bool:
    """Five wrong passwords for one email within ten minutes locks sign-in for that email."""
    recent = [t for t in _failures.get(email, []) if time.time() - t < 600]
    _failures[email] = recent
    return len(recent) >= 5


def record_failure(email: str):
    _failures.setdefault(email, []).append(time.time())


# ------------------------------------------------------------------ used by the routes
def get_db():
    with Session() as db:
        yield db


def current_user(authorization: str = Header(default=""), db=Depends(get_db)) -> User:
    """Read the token from the request and return the signed-in user, or refuse with 401."""
    try:
        payload = jwt.decode(authorization.removeprefix("Bearer ").strip(), SECRET, algorithms=["HS256"])
        user = db.get(User, int(payload["sub"]))
        if user and (not user.active or payload.get("pw") != password_stamp(user)):
            user = None                                   # disabled, or the password was reset since signing in
    except (jwt.PyJWTError, KeyError, ValueError):
        user = None
    if not user:
        raise HTTPException(401, "Please sign in again.")
    return user


def require(*roles):
    """Use as a dependency to allow only some roles, e.g. Depends(require("admin"))."""
    def check(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "Your account cannot do this.")
        return user
    return check


def can_see_patient(db, user: User, patient_id: int) -> bool:
    """The one rule for patient data:
       a patient sees only their own; a clinician only patients who shared with them; an admin none."""
    if user.role == "patient":
        return user.id == patient_id
    if user.role == "clinician":
        return db.query(Access).filter_by(patient_id=patient_id, clinician_id=user.id).first() is not None
    return False
