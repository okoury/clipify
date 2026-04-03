from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio
import hashlib
import shutil
import subprocess
import sys
import os
import tempfile
import json
import secrets
from datetime import datetime, timedelta

import database as db
import auth

# Always resolve paths relative to this file, regardless of cwd
HERE      = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(HERE, "clips")
PIPELINE  = os.path.join(HERE, "clipify-pipeline.py")

os.makedirs(CLIPS_DIR, exist_ok=True)
db.init_db()

# Limit concurrent pipeline subprocesses to avoid resource exhaustion
_PIPELINE_SEM = asyncio.Semaphore(2)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")


# ── Auth endpoints ─────────────────────────────────────────────────────────────

class RegisterBody(BaseModel):
    email: str
    name: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class UpdateProfileBody(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    default_annotate: Optional[bool] = None
    default_highlight: Optional[bool] = None
    default_emojis: Optional[bool] = None


class ChangePasswordBody(BaseModel):
    current_password: str
    new_password: str


class ForgotPasswordBody(BaseModel):
    email: str


class ResetPasswordBody(BaseModel):
    token: str
    new_password: str


@app.post("/api/auth/register")
def register(body: RegisterBody):
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if not body.name.strip():
        raise HTTPException(400, "Name is required")
    if "@" not in body.email:
        raise HTTPException(400, "Invalid email address")

    if db.get_user_by_email(body.email):
        raise HTTPException(400, "An account with that email already exists")

    password_hash = auth.hash_password(body.password)
    user_id = db.create_user(body.email, body.name, password_hash)
    user = db.get_user_by_id(user_id)
    token = auth.create_token(user_id)
    return {"token": token, "user": _user_dict(user)}


@app.post("/api/auth/login")
def login(body: LoginBody):
    user = db.get_user_by_email(body.email)
    if not user or not auth.verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    token = auth.create_token(user["id"])
    return {"token": token, "user": _user_dict(user)}


@app.get("/api/auth/me")
def get_me(user_id: int = Depends(auth.get_current_user_id)):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    stats = db.get_user_stats(user_id)
    return {**_user_dict(user), "stats": stats}


@app.put("/api/auth/me")
def update_me(body: UpdateProfileBody, user_id: int = Depends(auth.get_current_user_id)):
    fields = {}
    if body.name is not None:
        if not body.name.strip():
            raise HTTPException(400, "Name cannot be empty")
        fields["name"] = body.name.strip()
    if body.email is not None:
        if "@" not in body.email:
            raise HTTPException(400, "Invalid email address")
        existing = db.get_user_by_email(body.email)
        if existing and existing["id"] != user_id:
            raise HTTPException(400, "Email already in use")
        fields["email"] = body.email.lower().strip()
    if body.default_annotate is not None:
        fields["default_annotate"] = int(body.default_annotate)
    if body.default_highlight is not None:
        fields["default_highlight"] = int(body.default_highlight)
    if body.default_emojis is not None:
        fields["default_emojis"] = int(body.default_emojis)

    db.update_user(user_id, **fields)
    user = db.get_user_by_id(user_id)
    return _user_dict(user)


@app.put("/api/auth/password")
def change_password(body: ChangePasswordBody, user_id: int = Depends(auth.get_current_user_id)):
    user = db.get_user_by_id(user_id)
    if not auth.verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(400, "Current password is incorrect")
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters")
    db.update_user(user_id, password_hash=auth.hash_password(body.new_password))
    return {"ok": True}


@app.post("/api/auth/forgot-password")
def forgot_password(body: ForgotPasswordBody):
    user = db.get_user_by_email(body.email)
    # Always return the same message to avoid user enumeration
    if not user:
        return {"message": "If that email exists, a reset token has been printed to the server console."}

    token      = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(minutes=15)).isoformat()
    db.create_reset_token(user["id"], token, expires_at)

    # In production this would send an email; for local use, print to terminal
    print(f"\n{'='*56}", flush=True)
    print(f"  PASSWORD RESET TOKEN for {user['email']}", flush=True)
    print(f"  Token   : {token}", flush=True)
    print(f"  Expires : {expires_at} UTC (15 min)", flush=True)
    print(f"{'='*56}\n", flush=True)

    return {"message": "If that email exists, a reset token has been printed to the server console."}


@app.post("/api/auth/reset-password")
def reset_password(body: ResetPasswordBody):
    if len(body.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    row = db.get_reset_token(body.token)
    if not row:
        raise HTTPException(400, "Invalid or already-used reset token")

    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        raise HTTPException(400, "Reset token has expired — please request a new one")

    db.consume_reset_token(body.token)
    db.update_user(row["user_id"], password_hash=auth.hash_password(body.new_password))
    return {"message": "Password updated successfully"}


# ── History endpoints ──────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history(user_id: int = Depends(auth.get_current_user_id)):
    items = db.get_history(user_id)
    for item in items:
        if item.get("clips_json"):
            item["clips"] = json.loads(item["clips_json"])
        item.pop("clips_json", None)
    return {"history": items}


@app.delete("/api/history/{item_id}")
def delete_history(item_id: int, user_id: int = Depends(auth.get_current_user_id)):
    deleted = db.delete_history_item(item_id, user_id)
    if not deleted:
        raise HTTPException(404, "History item not found")
    return {"ok": True}


# ── Process endpoint ───────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    annotate: bool = Form(False),
    annotate_only: bool = Form(False),
    clip_length: str = Form("medium"),
    max_clips: int = Form(0),
    header_enabled: bool = Form(True),
    custom_hook: str = Form(""),
    user_id: Optional[int] = Depends(auth.get_optional_user_id),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    original_filename = file.filename or "video"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        input_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    output_fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(output_fd)

    try:
        clip_length  = clip_length if clip_length in ("short", "medium", "long") else "medium"
        max_clips    = max(0, min(int(max_clips), 20))
        custom_hook  = (custom_hook or "").strip()[:120]   # cap length, strip whitespace
        cache_mode   = "annotate_only" if annotate_only else ("annotate" if annotate else "clips")

        # ── Cache lookup ──────────────────────────────────────────────────────
        hasher = hashlib.sha256()
        with open(input_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

        # Skip cache when custom_hook is set (title/summary would differ)
        cached = db.get_cache(file_hash, cache_mode, clip_length, max_clips) if not custom_hook else None
        if cached is not None:
            if user_id:
                clips = cached.get("clips", [])
                db.add_history(
                    user_id=user_id,
                    filename=original_filename,
                    mode="annotate" if annotate_only else "clips",
                    clips_count=len(clips),
                    clips_json=json.dumps(clips),
                )
            return cached

        # ── Run pipeline (max 2 concurrent) ──────────────────────────────────
        cmd = [sys.executable, PIPELINE, input_path, output_path]
        mode_str = "annotate_only" if annotate_only else ("annotate" if annotate else "")
        cmd.extend([mode_str, clip_length, str(max_clips),
                    "1" if header_enabled else "0",
                    custom_hook])

        async with _PIPELINE_SEM:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(cmd, capture_output=True, text=True, cwd=HERE, timeout=1800),
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=504,
                    detail="Processing timed out after 30 minutes. Try a shorter video.",
                )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error: {stderr or 'unknown error (no stderr)'}",
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Pipeline produced no output.")

        with open(output_path) as f:
            data = json.load(f)

        # ── Save to cache ─────────────────────────────────────────────────────
        db.set_cache(file_hash, cache_mode, clip_length, max_clips, data)

        # ── Save to history if authenticated ──────────────────────────────────
        if user_id:
            clips = data.get("clips", [])
            db.add_history(
                user_id=user_id,
                filename=original_filename,
                mode="annotate" if annotate_only else "clips",
                clips_count=len(clips),
                clips_json=json.dumps(clips),
            )

        return data

    finally:
        for path in (input_path, output_path):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _user_dict(user) -> dict:
    return {
        "id":               user["id"],
        "email":            user["email"],
        "name":             user["name"],
        "created_at":       user["created_at"],
        "default_annotate": bool(user["default_annotate"]),
        "default_highlight": bool(user["default_highlight"]),
        "default_emojis":   bool(user["default_emojis"]),
    }
