"""StoolCheck AI - FastAPI backend (Phase 1).

Endpoints:
    GET  /health          -> liveness check
    POST /register        -> create an account, returns a JWT (201)
    POST /login           -> exchange credentials for a JWT
    POST /analyze  [JWT]  -> upload image, run the SELECTED model, save result
    POST /upload   [JWT]  -> alias of /analyze (same handler, same contract)
    GET  /history  [JWT]  -> the logged-in user's data for the dashboard
    GET  /evaluation      -> YOLOv8 training metrics (public)
    GET  /                -> serves the frontend (frontend/index.html)

MULTI-MODEL (Phase 6):
    The upload endpoints accept a `model_choice` form field:
        "yolov8"      (default) - robust YOLOv8n-cls checkpoint (best.pt)
        "mobilenetv3"           - lightweight MobileNetV3 (mock until trained)

PRIVACY-FIRST CONTRACT:
    Uploaded images are written to a temporary file, used for inference,
    and DELETED in a `finally` block via os.remove() so they never survive
    the request - even if inference or the DB write fails.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai_mock import analyze_stool_image  # noqa: F401  (replaced below by Phase 3)
from auth import create_access_token, get_current_user, hash_password, verify_password
from database import Base, engine, get_db
from evaluation import load_evaluation
from inference import MODEL_CHOICES, analyze_stool_image  # Phase 3/6: YOLOv8 + MobileNetV3 (mock fallbacks)
from models import Analysis, User

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Verify DB connectivity (MySQL/XAMPP) and ensure tables exist."""
    try:
        Base.metadata.create_all(bind=engine)
        from sqlalchemy import inspect as _sa_inspect

        _inspector = _sa_inspect(engine)
        if "analyses" in _inspector.get_table_names():
            _cols = {c["name"]: c for c in _inspector.get_columns("analyses")}
            if "INT" not in str(_cols.get("user_id", {}).get("type", "")).upper():
                print("[DB] WARNING: legacy schema (no user FK) -> run:  python init_db.py")
        print(f"[DB] Connected: {engine.url.render_as_string(hide_password=True)}")
    except Exception as exc:
        print(f"[DB] WARNING: database unavailable: {exc}")
        print("[DB] Start XAMPP MySQL, then run:  python init_db.py")
    yield


app = FastAPI(
    title="StoolCheck AI API",
    description="Bristol Stool Scale analysis with privacy-first image handling.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the static frontend (Phase 2) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
SUPPORTED_MODELS = set(MODEL_CHOICES)  # {"yolov8", "mobilenetv3"}


# --- Phase 5: auth schemas ---------------------------------------------------
class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=1, max_length=128)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "stoolcheck-ai"}


@app.post("/register", status_code=201)
def register(payload: RegisterIn, db: Session = Depends(get_db)) -> dict:
    """Create an account (email + display username + password), return a JWT."""
    email = payload.email.lower()
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=409, detail="Email already registered.")
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already taken.")
    user = User(
        username=payload.username,
        email=email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "email": user.email},
    }


@app.post("/login")
def login(payload: LoginIn, db: Session = Depends(get_db)) -> dict:
    """Verify email + password and return a JWT."""
    user = (
        db.query(User)
        .filter(func.lower(User.email) == payload.email.lower())
        .first()
    )
    if user is None or not verify_password(payload.password, user.hashed_password):
        # Same answer for unknown email and wrong password (no enumeration).
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "email": user.email},
    }


@app.post("/analyze")
@app.post("/upload")  # explicit alias: same handler, same privacy contract
def analyze(
    file: UploadFile = File(..., description="Stool image (JPG/PNG/WEBP/BMP/GIF)"),
    model_choice: str = Form(
        "yolov8",
        description="Inference engine: 'yolov8' (robust) or 'mobilenetv3' (lightweight)",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Run the SELECTED model on an uploaded image, then DELETE the image.

    Identity comes from the JWT (get_current_user) - the client cannot
    choose which user_id gets stored.
    """
    model_choice = (model_choice or "yolov8").strip().lower()
    if model_choice not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown model_choice '{model_choice}'. "
                f"Valid options: {sorted(SUPPORTED_MODELS)}."
            ),
        )
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type. Allowed: {sorted(ALLOWED_TYPES)}",
        )

    # Read once up-front so we can enforce a size cap before touching disk.
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    ext = os.path.splitext(file.filename or "")[1].lower() or ".img"
    tmp_path: str | None = None

    try:
        # 1) Save the image to a temporary file (no permanent storage).
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # 2) Run AI inference with the chosen engine. YOLOv8 keeps the legacy
        #    single-argument call (the default path); only MobileNetV3 passes
        #    the extra keyword, so existing integrations/tests are unaffected.
        if model_choice == "yolov8":
            result = analyze_stool_image(tmp_path)
        else:
            result = analyze_stool_image(tmp_path, model_choice=model_choice)

        # 3) Persist ONLY user_id, timestamp, bristol_type.
        record = Analysis(
            user_id=current_user.id,
            bristol_type=result["bristol_type"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except HTTPException:
        db.rollback()
        raise
    except ValueError as exc:
        # Corrupt/unreadable image or unusable model output -> client error.
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        # 4) PRIVACY: always delete the image, success or failure.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                # Last-resort: truncate so no pixels remain even if unlink fails.
                try:
                    with open(tmp_path, "wb") as f:
                        f.truncate(0)
                except OSError:
                    pass

    image_deleted = tmp_path is None or not os.path.exists(tmp_path)
    if not image_deleted:
        raise HTTPException(status_code=500, detail="Privacy violation: image retained.")

    return {
        "id": record.id,
        "user_id": record.user_id,
        "username": current_user.username,
        "timestamp": record.timestamp.isoformat(),
        "bristol_type": result["bristol_type"],
        "label": result["label"],
        "description": result["description"],
        "recommendations": result["recommendations"],
        "confidence": result["confidence"],
        "model": result["model"],
        "model_choice": model_choice,
        "image_deleted": image_deleted,
    }


@app.get("/history")
def history(
    current_user: User = Depends(get_current_user),
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return the logged-in user's history (oldest first) for the dashboard."""
    limit = max(1, min(limit, 1000))
    rows = (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.timestamp.asc(), Analysis.id.asc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "bristol_type": r.bristol_type,
        }
        for r in rows
    ]


@app.get("/evaluation")
def evaluation() -> dict:
    """YOLOv8 training metrics for the Model Evaluation section (Phase 4)."""
    return load_evaluation()


# --- Phase 2: serve the static frontend at "/" -----------------------------
# Registered LAST so API routes above (/health, /analyze, /history) win matching.


class SafeStaticFiles(StaticFiles):
    """StaticFiles hardened against malformed request paths.

    On Windows, os.path.realpath() raises OSError (WinError 123) for paths
    containing characters like a mid-path ':' (e.g. a bogus request such as
    GET /)http://...). Starlette re-raises those as 500s with a traceback;
    this subclass converts them into a clean 404 instead.
    """

    def lookup_path(self, path: str):
        try:
            return super().lookup_path(path)
        except (OSError, ValueError):
            return "", None


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", SafeStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
