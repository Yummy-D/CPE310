"""JWT authentication for StoolCheck AI.

- Passwords: bcrypt via passlib (bcrypt pinned <4.1 for passlib 1.7.4 compat).
- Tokens: PyJWT HS256, default 12h expiry; claims: sub=email (login id), uid.
- Secrets: read SECRET_KEY (legacy alias: STOOLCHECK_SECRET_KEY) from the
  environment / .env — see .env.example.  When ENVIRONMENT=production the app
  REFUSES to start with a missing or default secret (a guessable secret would
  let anyone forge tokens).
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import IS_PRODUCTION, env
from database import get_db
from models import User

_DEV_SECRET = "dev-only-change-me-stoolcheck-ai"

SECRET_KEY = env("SECRET_KEY") or env("STOOLCHECK_SECRET_KEY") or _DEV_SECRET
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(env("JWT_EXPIRE_MINUTES", str(12 * 60)) or 720)

if IS_PRODUCTION and (SECRET_KEY == _DEV_SECRET or len(SECRET_KEY) < 32):
    raise SystemExit(
        "[FATAL] SECRET_KEY is missing or still the development default while "
        "ENVIRONMENT=production.\n"
        "        Set a strong random secret, e.g.:\n"
        "          python -c \"import secrets; print(secrets.token_urlsafe(64))\""
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Extracts the "Authorization: Bearer <token>" header and, when it is missing,
# answers 401 automatically (also powers the /docs "Authorize" button).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        # Malformed hash (e.g. migrated placeholder rows) -> never matches.
        return False


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.email,  # login identifier
        "uid": user.id,     # immutable - used to resolve the user
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: validate the Bearer JWT and load the user.

    The user_id used by /analyze and /history ALWAYS comes from here -
    never from a client-controlled payload.
    """
    credentials_error = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("uid")
        email = payload.get("sub")
        if uid is None and not email:
            raise credentials_error
    except jwt.PyJWTError:
        raise credentials_error

    # Resolve by immutable user id first; fall back to the email claim.
    user = None
    if isinstance(uid, int):
        user = db.query(User).filter(User.id == uid).first()
    if user is None and isinstance(email, str) and email:
        user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_error
    return user
