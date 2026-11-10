#!/usr/bin/env python3
"""
Video Studio: a small local web app on top of make_video.py.

  pip install -r requirements.txt
  python app.py            # then open http://127.0.0.1:5000

Reads GROQ_API_KEY, GROQ_MODEL and (optional) PIPER_VOICE from your .env file.
Videos are saved in ./videos/<id>/video.mp4 with a meta.json next to each one.
"""

import hmac
import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

import llm
import make_video as mv  # also loads .env

BASE = Path(__file__).resolve().parent
VIDEOS = BASE / "videos"
VIDEOS.mkdir(exist_ok=True)

PIPER_VOICE = os.getenv("PIPER_VOICE") or None
MAX_SCRIPT_CHARS = 20_000
MAX_SCENES = int(os.getenv("MAX_SCENES", "30"))
MAX_VIDEOS = int(os.getenv("MAX_VIDEOS", "0"))  # keep only the newest N videos (0 = keep all)
APP_PASSWORD = os.getenv("APP_PASSWORD") or None
# Letting visitors type any base URL makes the server call it, so it is off by default on Render.
ALLOW_CUSTOM_BASE_URL = os.getenv("ALLOW_CUSTOM_BASE_URL", "0" if os.getenv("RENDER") else "1") == "1"

if os.getenv("RENDER") and not APP_PASSWORD:  # RENDER is set automatically on Render
    raise RuntimeError("Set APP_PASSWORD in the Environment settings: the app has no other login, "
                       "and a public URL would let anyone use your Groq key and server.")
ID_RE = re.compile(r"^[0-9a-f]{12}$")

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=1)  # one render at a time keeps your CPU usable
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ---------- password protection (set APP_PASSWORD to enable) ----------

@app.before_request
def require_password():
    if not APP_PASSWORD or request.path == "/healthz":
        return None
    auth = request.authorization  # browser shows a login box; username can be anything
    if auth and hmac.compare_digest((auth.password or "").encode(), APP_PASSWORD.encode()):
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Video Studio"'})


@app.get("/healthz")
def healthz():
    return "ok"


# ---------- helpers ----------

def prune_videos() -> None:
    """Delete the oldest videos beyond MAX_VIDEOS (the disk is small and temporary on free hosts)."""
    if MAX_VIDEOS <= 0:
        return
    found = []
    for folder in VIDEOS.iterdir():
        meta = read_meta(folder) if folder.is_dir() else None
        if meta:
            found.append((meta["created"], folder))
    found.sort(reverse=True)
    for _, folder in found[MAX_VIDEOS:]:
        shutil.rmtree(folder, ignore_errors=True)


def voice_available() -> bool:
    return bool(PIPER_VOICE and Path(PIPER_VOICE).exists() and shutil.which("piper"))


def resolve_llm(data: dict, need_model: bool = True) -> "llm.LLMConfig":
    """Build an LLM config from the request ('llm' object) with .env fallbacks. Raises ValueError."""
    spec = data.get("llm") or {}
    return llm.build_config(
        spec.get("provider") or llm.default_provider(),
        model=spec.get("model"), api_key=spec.get("api_key"), base_url=spec.get("base_url"),
        allow_custom_base=ALLOW_CUSTOM_BASE_URL, need_model=need_model,
    )


def valid_id(video_id: str) -> Path:
    if not ID_RE.match(video_id):
        abort(404)
    folder = VIDEOS / video_id
    if not folder.is_dir():
        abort(404)
    return folder


def read_meta(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def set_job(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


# ---------- background build ----------

def run_job(job_id: str, script_text: str, title: str, use_voice: bool, cfg: "llm.LLMConfig | None") -> None:
    folder = VIDEOS / job_id
    try:
        set_job(job_id, status="running", message="Reading script...")
        scenes = mv.parse_script_text(script_text)
        if not scenes:
            raise ValueError("No scenes found in the script.")
        if len(scenes) > MAX_SCENES:
            raise ValueError(f"Too many scenes ({len(scenes)}). Maximum is {MAX_SCENES}.")
        if any(not sc["narration"].strip() for sc in scenes):
            raise ValueError("Every scene needs some narration text (the lines under the bullets).")

        if any(not sc["title"] or not sc["bullets"] for sc in scenes):
            set_job(job_id, message="Writing slide titles and bullets...")
            mv.fill_missing_slides(scenes, cfg)

        def on_progress(frac: float, msg: str) -> None:
            set_job(job_id, progress=round(frac, 3), message=msg)

        duration = mv.build_video(
            scenes,
            PIPER_VOICE if (use_voice and voice_available()) else None,
            folder / "work",
            folder / "video.mp4",
            on_progress=on_progress,
            quiet=True,
        )
        shutil.rmtree(folder / "work", ignore_errors=True)

        meta = {
            "id": job_id,
            "title": title or scenes[0]["title"] or "Untitled video",
            "created": time.time(),
            "duration": round(duration, 1),
            "scenes": len(scenes),
            "voice": bool(use_voice and voice_available()),
            "script": mv.scenes_to_text(scenes),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        prune_videos()
        set_job(job_id, status="done", progress=1.0, message="Done", meta=meta)
    except Exception as e:  # report any failure to the UI
        shutil.rmtree(folder, ignore_errors=True)
        set_job(job_id, status="error", error=str(e), message="Failed")


# ---------- routes ----------

@app.get("/")
def index():
    # index.html can live in ./templates/ or right next to app.py
    for folder in (BASE / "templates", BASE):
        if (folder / "index.html").exists():
            return send_from_directory(folder, "index.html")
    return (f"index.html not found. Put it in {BASE / 'templates'} or next to app.py.", 500)


@app.get("/api/config")
def config():
    providers = []
    for pid, spec in llm.PROVIDERS.items():
        providers.append({
            "id": pid, "label": spec["label"],
            "server_key": bool(llm.server_key(pid)),
            "key_optional": bool(spec.get("key_optional")),
            "default_model": llm.default_model(pid),
            "needs_base_url": pid == "custom" and ALLOW_CUSTOM_BASE_URL,
            "available": pid != "custom" or ALLOW_CUSTOM_BASE_URL or bool(llm.env_base_url("custom")),
        })
    return jsonify(providers=providers, default_provider=llm.default_provider(), voice=voice_available())


@app.post("/api/models")
def models():
    data = request.get_json(silent=True) or {}
    try:
        cfg = resolve_llm(data, need_model=False)
        return jsonify(models=llm.list_models(cfg))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except llm.LLMError as e:
        return jsonify(error=str(e)), 502


@app.post("/api/generate-script")
def generate_script():
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify(error="Enter a topic first."), 400
    try:
        count = max(2, min(int(data.get("scenes", 5)), 12))
    except (TypeError, ValueError):
        count = 5
    try:
        cfg = resolve_llm(data)
        scenes = mv.generate_scenes(topic[:300], count, cfg)
        return jsonify(script=mv.scenes_to_text(scenes), title=topic[:80])
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except llm.LLMError as e:
        return jsonify(error=str(e)), 502


@app.post("/api/build")
def build():
    data = request.get_json(silent=True) or {}
    script = (data.get("script") or "").strip()
    if not script:
        return jsonify(error="The script is empty."), 400
    if len(script) > MAX_SCRIPT_CHARS:
        return jsonify(error=f"Script is too long (max {MAX_SCRIPT_CHARS} characters)."), 400

    job_id = uuid.uuid4().hex[:12]
    (VIDEOS / job_id).mkdir(parents=True)
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "Queued...", "error": None}
    try:
        cfg = resolve_llm(data)  # optional: only used to write slide titles/bullets
    except ValueError:
        cfg = None
    executor.submit(run_job, job_id, script, (data.get("title") or "").strip()[:100], bool(data.get("voice")), cfg)
    return jsonify(job=job_id), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    if not ID_RE.match(job_id):
        abort(404)
    with jobs_lock:
        job = dict(jobs.get(job_id) or {})
    if not job:
        abort(404)
    return jsonify(job)


@app.get("/api/videos")
def list_videos():
    items = []
    for folder in VIDEOS.iterdir():
        meta = read_meta(folder) if folder.is_dir() else None
        if meta and (folder / "video.mp4").exists():
            items.append({k: meta[k] for k in ("id", "title", "created", "duration", "scenes", "voice")})
    items.sort(key=lambda m: m["created"], reverse=True)
    return jsonify(items)


@app.get("/api/videos/<video_id>")
def get_video(video_id):
    meta = read_meta(valid_id(video_id))
    if not meta:
        abort(404)
    return jsonify(meta)


@app.get("/videos/<video_id>.mp4")
def serve_video(video_id):
    folder = valid_id(video_id)
    meta = read_meta(folder) or {}
    name = re.sub(r"[^\w\- ]+", "", meta.get("title", "video")).strip().replace(" ", "_") or "video"
    return send_from_directory(
        folder, "video.mp4", mimetype="video/mp4", conditional=True,
        as_attachment=request.args.get("download") == "1", download_name=f"{name}.mp4",
    )


@app.delete("/api/videos/<video_id>")
def delete_video(video_id):
    shutil.rmtree(valid_id(video_id), ignore_errors=True)
    return jsonify(ok=True)


if __name__ == "__main__":
    print("Video Studio running at http://127.0.0.1:5000")
    ready = [spec["label"] for pid, spec in llm.PROVIDERS.items() if llm.server_key(pid)]
    print(f"  AI providers with a key in .env: {', '.join(ready) if ready else 'none (users can paste a key in the UI)'}")
    print(f"  Voiceover: {'Piper ready' if voice_available() else 'off (set PIPER_VOICE in .env to enable)'}")
    app.run(host="127.0.0.1", port=5000, threaded=True)