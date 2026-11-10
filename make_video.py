#!/usr/bin/env python3
"""
Topic -> explainer video, using Groq (free tier) for the script and local tools for the rest.

Pipeline:
  1. An LLM (Groq, OpenAI, Anthropic or any OpenAI-compatible API) writes a scene-by-scene script as JSON
  2. Piper (optional, local TTS) turns narration into audio. Without Piper, video is silent
     and each scene's length is estimated from the word count.
  3. Pillow draws a slide image per scene
  4. ffmpeg turns the slides + audio into an MP4

Setup:
  pip install requests imageio-ffmpeg pillow python-dotenv
  export GROQ_API_KEY="your_key"          # or OPENAI_API_KEY / ANTHROPIC_API_KEY, see .env.example
  (optional) pip install piper-tts  and download a voice, e.g. en_US-lessac-medium.onnx

Usage:
  python make_video.py "How does DNS work?"
  python make_video.py "How does DNS work?" --provider anthropic --model claude-haiku-5-5
  python make_video.py "How does DNS work?" --voice ./en_US-lessac-medium.onnx
  python make_video.py --script my_script.txt      # your own script (see parse_script for formats)
  python make_video.py --script build/scenes.json  # re-render after editing a saved script
  python make_video.py --demo                      # no Groq needed, uses built-in sample scenes
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import llm

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads GROQ_API_KEY / GROQ_MODEL from a .env file next to where you run the script
except ImportError:
    pass  # fine if you export the variables in your shell instead

W, H = 1920, 1080                                    # slides are always drawn at 1080p
FPS = int(os.getenv("VIDEO_FPS", "15"))               # slides are still images, so low fps is fine; lower = faster
OUT_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1080"))   # output height, e.g. 720 or 480 on weak servers
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "veryfast")  # x264 speed preset: ultrafast (fastest) ... medium (smallest files)
VIDEO_THREADS = int(os.getenv("VIDEO_THREADS", "2"))

PALETTES = [
    ((15, 23, 42), (56, 189, 248)),    # navy / sky
    ((24, 24, 27), (250, 204, 21)),    # charcoal / yellow
    ((17, 24, 39), (52, 211, 153)),    # slate / green
    ((30, 20, 50), (244, 114, 182)),   # purple / pink
]

DEMO_SCENES = [
    {"title": "What is DNS?",
     "bullets": ["The internet's phone book", "Turns names into IP addresses"],
     "narration": "DNS is the phone book of the internet. It turns names like example dot com into IP addresses."},
    {"title": "The lookup steps",
     "bullets": ["Browser checks its cache", "Resolver asks root, TLD, then authoritative server", "IP address returned"],
     "narration": "Your browser checks its cache first. If needed, a resolver asks the root, then the top level domain, then the authoritative server."},
    {"title": "Why it matters",
     "bullets": ["Fast sites need fast DNS", "DNS problems look like outages"],
     "narration": "Fast DNS makes sites feel fast, and when DNS breaks, it often looks like the whole internet is down."},
]


# ---------- 1. Script generation (any LLM provider) ----------

def generate_scenes(topic: str, num_scenes: int, cfg: "llm.LLMConfig") -> list[dict]:
    system = (
        "You write scripts for short YouTube tech explainer videos. "
        "Reply with JSON only, in this exact shape: "
        '{"scenes": [{"title": str, "bullets": [str, ...], "narration": str}]}. '
        "Titles: max 6 words. 2-4 bullets per scene, each max 10 words. "
        "Narration: 2-3 plain sentences, written to be spoken aloud, no markdown."
    )
    user = f"Topic: {topic}\nNumber of scenes: {num_scenes}\nAudience: curious beginners."
    scenes = llm.chat_json(cfg, system, user).get("scenes", [])
    if not scenes:
        raise llm.LLMError("The model returned no scenes. Try again or pick another model.")
    return scenes


# ---------- 1b. Custom script (your own text) ----------

def parse_script(path: Path) -> list[dict]:
    """
    Load scenes from your own file.

    .json  -> a list of scenes, or {"scenes": [...]} (same format as build/scenes.json)
    .txt/.md, structured format:
        # Scene title
        - bullet one
        - bullet two
        Narration text, can span several lines.
    .txt/.md, plain format (no '# ' headings): each paragraph (blank-line separated)
        becomes one scene's narration; title/bullets are filled in automatically.
    """
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        scenes = data["scenes"] if isinstance(data, dict) else data
        return [{"title": sc.get("title", ""), "bullets": sc.get("bullets", []),
                 "narration": sc.get("narration", "")} for sc in scenes]

    return parse_script_text(text)


def parse_script_text(text: str) -> list[dict]:
    """Parse structured ('# Title' / '- bullet' / narration) or plain-paragraph script text."""
    text = text.strip()
    lines = text.splitlines()
    if any(l.startswith("# ") for l in lines):
        scenes, cur = [], None
        for line in lines:
            if line.startswith("# "):
                cur = {"title": line[2:].strip(), "bullets": [], "narration": ""}
                scenes.append(cur)
            elif cur is None or not line.strip():
                continue
            elif line.lstrip().startswith(("- ", "* ")):
                cur["bullets"].append(line.lstrip()[2:].strip())
            else:
                cur["narration"] = (cur["narration"] + " " + line.strip()).strip()
        return scenes

    paras = [p.strip().replace("\n", " ") for p in text.split("\n\n") if p.strip()]
    return [{"title": "", "bullets": [], "narration": p} for p in paras]


def scenes_to_text(scenes: list[dict]) -> str:
    """Turn scenes into the editable structured text format."""
    blocks = []
    for sc in scenes:
        lines = [f"# {sc.get('title', '').strip()}"]
        lines += [f"- {b}" for b in sc.get("bullets", [])]
        lines.append(sc.get("narration", "").strip())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _short(sentence: str, max_words: int) -> str:
    words = sentence.strip().rstrip(".").split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def heuristic_slide(scene: dict, i: int) -> None:
    """Offline fallback: build a title/bullets from the narration itself."""
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", scene["narration"]) if x.strip()]
    if not scene["title"]:
        scene["title"] = _short(sentences[0], 6) if sentences else f"Part {i + 1}"
    if not scene["bullets"]:
        scene["bullets"] = [_short(x, 10) for x in sentences[1:4]]


def fill_missing_slides(scenes: list[dict], cfg: "llm.LLMConfig | None") -> None:
    """Fill in missing titles/bullets, using the chosen LLM when one is configured."""
    todo = [i for i, sc in enumerate(scenes) if not sc["title"] or not sc["bullets"]]
    if not todo:
        return
    if cfg is not None:
        try:
            system = (
                "You create slide text for a tech explainer video. You get numbered narration "
                'passages. Reply with JSON only: {"slides": [{"title": str, "bullets": [str, ...]}]} '
                "with exactly one entry per passage, in order. Titles max 6 words; 2-4 bullets, "
                "each max 10 words, summarizing the passage."
            )
            user = "\n\n".join(f"{n + 1}. {scenes[i]['narration']}" for n, i in enumerate(todo))
            slides = llm.chat_json(cfg, system, user).get("slides", [])
            if len(slides) == len(todo):
                for i, sl in zip(todo, slides):
                    scenes[i]["title"] = scenes[i]["title"] or sl.get("title", "")
                    scenes[i]["bullets"] = scenes[i]["bullets"] or sl.get("bullets", [])
        except Exception as e:  # fall back to offline heuristic
            print(f"  ! AI slide generation failed ({e}); using simple fallback", file=sys.stderr)
    for i in todo:
        heuristic_slide(scenes[i], i)


# ---------- 2. Voiceover (Piper, optional) ----------

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synthesize(text: str, out_wav: Path, voice_model: str | None) -> float | None:
    """Return audio duration in seconds, or None if no TTS is available."""
    if not voice_model:
        return None
    piper = shutil.which("piper")
    if not piper:
        print("  ! 'piper' not found on PATH, skipping voiceover", file=sys.stderr)
        return None
    subprocess.run(
        [piper, "--model", voice_model, "--output_file", str(out_wav)],
        input=text.encode("utf-8"), check=True, capture_output=True,
    )
    return wav_duration(out_wav)


def estimate_duration(text: str) -> float:
    words = len(text.split())
    return max(4.0, words / 2.6)  # ~155 words per minute


# ---------- 3. Slide drawing (Pillow) ----------

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_slide(scene: dict, index: int, total: int, out_png: Path) -> None:
    bg, accent = PALETTES[index % len(PALETTES)]
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    title_font = load_font(96, bold=True)
    bullet_font = load_font(54)
    small_font = load_font(32)

    # accent bar + title
    d.rectangle([100, 120, 116, 250], fill=accent)
    title = "\n".join(textwrap.wrap(scene.get("title", ""), width=32)[:2])
    d.multiline_text((150, 110), title, font=title_font, fill="white", spacing=10)

    # bullets
    y = 400
    for b in scene.get("bullets", [])[:4]:
        d.ellipse([150, y + 18, 174, y + 42], fill=accent)
        wrapped = textwrap.wrap(b, width=46)[:2]
        d.multiline_text((210, y), "\n".join(wrapped), font=bullet_font, fill=(226, 232, 240), spacing=8)
        y += 90 * len(wrapped) + 40

    # footer progress
    d.text((100, H - 80), f"{index + 1} / {total}", font=small_font, fill=(148, 163, 184))
    bar_w = W - 200
    d.rectangle([100, H - 30, 100 + bar_w, H - 22], fill=(51, 65, 85))
    d.rectangle([100, H - 30, 100 + int(bar_w * (index + 1) / total), H - 22], fill=accent)

    if OUT_HEIGHT != H:  # downscale for faster rendering (x264 needs even dimensions)
        out_w = round(OUT_HEIGHT * W / H) // 2 * 2
        img = img.resize((out_w, OUT_HEIGHT // 2 * 2), Image.LANCZOS)
    img.save(out_png)


# ---------- 4. Assemble (ffmpeg) ----------

def ffmpeg_exe() -> str:
    """Use the ffmpeg bundled with imageio-ffmpeg, else one on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg")


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *args],
                       capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-600:])


def build_video(scenes, voice_model, workdir: Path, out_mp4: Path, on_progress=None, quiet=False) -> float:
    """
    Render scenes to out_mp4. on_progress(fraction 0-1, message) is optional. Returns duration in seconds.

    Each slide is a still image, so we hand it straight to ffmpeg (one short clip per scene, with
    fades), then join the clips without re-encoding. This is far faster than building every frame
    in Python.
    """
    def report(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(frac, msg)

    workdir.mkdir(parents=True, exist_ok=True)
    n = len(scenes)

    # Phase 1: slide images + voiceover audio
    prepared = []  # (png, wav or None, duration)
    for i, scene in enumerate(scenes):
        msg = f"Preparing scene {i + 1}/{n}: {scene.get('title')}"
        print(msg)
        report(0.15 * i / n, msg)
        png = workdir / f"slide_{i:02d}.png"
        wav = workdir / f"voice_{i:02d}.wav"
        draw_slide(scene, i, n, png)
        narration = scene.get("narration", "")
        audio_len = synthesize(narration, wav, voice_model)
        duration = (audio_len + 0.6) if audio_len else estimate_duration(narration)
        prepared.append((png, wav if audio_len else None, duration))

    has_audio = any(w for _, w, _ in prepared)

    # Phase 2: one clip per scene
    segments = []
    for i, (png, wav, dur) in enumerate(prepared):
        report(0.15 + 0.8 * i / n, f"Rendering scene {i + 1}/{n}...")
        seg = workdir / f"seg_{i:02d}.mp4"
        fade = min(0.4, dur / 4)
        args = ["-loop", "1", "-framerate", str(FPS), "-i", str(png)]
        if wav:
            args += ["-i", str(wav)]
        elif has_audio:  # keep every clip's streams identical so they can be joined
            args += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
        vf = f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={dur - fade:.3f}:d={fade:.2f},format=yuv420p"
        args += ["-vf", vf, "-t", f"{dur:.3f}", "-r", str(FPS),
                 "-c:v", "libx264", "-preset", VIDEO_PRESET, "-tune", "stillimage",
                 "-threads", str(VIDEO_THREADS)]
        if has_audio:
            args += ["-af", "apad", "-c:a", "aac", "-ar", "44100", "-ac", "2"]
        else:
            args += ["-an"]
        run_ffmpeg(args + [str(seg)])
        segments.append(seg)

    # Phase 3: join the clips (no re-encoding)
    report(0.95, "Joining scenes...")
    listing = workdir / "list.txt"
    listing.write_text("".join(f"file '{seg.name}'\n" for seg in segments), encoding="utf-8")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
                "-movflags", "+faststart", str(Path(out_mp4).resolve())], cwd=workdir)

    report(1.0, "Done")
    print(f"\nDone: {out_mp4}")
    return float(sum(d for _, _, d in prepared))


def main() -> None:
    ap = argparse.ArgumentParser(description="Topic -> explainer video using an AI model + ffmpeg")
    ap.add_argument("topic", nargs="?", help="Video topic, e.g. 'How does DNS work?'")
    ap.add_argument("--script", help="Use your own script file (.txt, .md or .json) instead of Groq writing it")
    ap.add_argument("--no-ai-slides", "--no-groq-slides", dest="no_ai_slides", action="store_true",
                    help="With --script: don't use an AI model for titles/bullets, use a simple offline fallback")
    ap.add_argument("--provider", choices=list(llm.PROVIDERS), help="AI provider (default: LLM_PROVIDER from .env, else groq)")
    ap.add_argument("--model", help="Model id (default: from .env, e.g. GROQ_MODEL)")
    ap.add_argument("--scenes", type=int, default=5, help="Number of scenes (default 5)")
    ap.add_argument("--voice", help="Path to a Piper .onnx voice model (optional)")
    ap.add_argument("--out", default="output.mp4", help="Output file name")
    ap.add_argument("--demo", action="store_true", help="Skip Groq, use built-in sample scenes")
    args = ap.parse_args()

    def make_cfg(required: bool):
        try:
            return llm.build_config(args.provider or llm.default_provider(), model=args.model)
        except ValueError as e:
            if required:
                sys.exit(str(e))
            return None

    if args.demo:
        scenes = DEMO_SCENES
    elif args.script:
        scenes = parse_script(Path(args.script))
        if not scenes:
            sys.exit("No scenes found in your script file.")
        fill_missing_slides(scenes, None if args.no_ai_slides else make_cfg(False))
        print(f"Loaded {len(scenes)} scenes from {args.script}")
    else:
        if not args.topic:
            ap.error("Provide a topic, --script FILE, or --demo")
        cfg = make_cfg(True)
        print(f"Asking {llm.PROVIDERS[cfg.provider]['label']} ({cfg.model}) for a script on: {args.topic}")
        scenes = generate_scenes(args.topic, args.scenes, cfg)

    workdir = Path("build")
    workdir.mkdir(exist_ok=True)
    (workdir / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")  # edit & reuse if you like

    build_video(scenes, args.voice, workdir, Path(args.out))


if __name__ == "__main__":
    main()