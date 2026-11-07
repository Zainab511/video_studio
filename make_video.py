#!/usr/bin/env python3
"""
Topic -> explainer video, using Groq (free tier) for the script and local tools for the rest.

Pipeline:
  1. Groq LLM writes a scene-by-scene script as JSON
  2. Piper (optional, local TTS) turns narration into audio. Without Piper, video is silent
     and each scene's length is estimated from the word count.
  3. Pillow draws a slide image per scene
  4. MoviePy joins slides + audio into an MP4

Setup:
  pip install groq moviepy pillow python-dotenv
  export GROQ_API_KEY="your_key"          # free key from console.groq.com
  (optional) pip install piper-tts  and download a voice, e.g. en_US-lessac-medium.onnx

Usage:
  python make_video.py "How does DNS work?"
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

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads GROQ_API_KEY / GROQ_MODEL from a .env file next to where you run the script
except ImportError:
    pass  # fine if you export the variables in your shell instead

W, H = 1920, 1080
FPS = 24
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")  # verify the name at console.groq.com/docs/models

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


# ---------- 1. Script generation (Groq) ----------

def groq_json(system: str, user: str) -> dict:
    """Call Groq and return the parsed JSON object from the reply."""
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from the environment
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    raw = resp.choices[0].message.content or ""
    # Reasoning models (e.g. Qwen) may prepend <think>...</think>; strip it and grab the JSON object
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Model did not return JSON. Raw response:\n" + raw)
    return json.loads(match.group(0))


def generate_scenes(topic: str, num_scenes: int) -> list[dict]:
    system = (
        "You write scripts for short YouTube tech explainer videos. "
        "Reply with JSON only, in this exact shape: "
        '{"scenes": [{"title": str, "bullets": [str, ...], "narration": str}]}. '
        "Titles: max 6 words. 2-4 bullets per scene, each max 10 words. "
        "Narration: 2-3 plain sentences, written to be spoken aloud, no markdown."
    )
    user = f"Topic: {topic}\nNumber of scenes: {num_scenes}\nAudience: curious beginners."
    scenes = groq_json(system, user).get("scenes", [])
    if not scenes:
        raise RuntimeError("Groq returned no scenes.")
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


def fill_missing_slides(scenes: list[dict], use_groq: bool) -> None:
    """Fill in missing titles/bullets, using Groq when possible."""
    todo = [i for i, sc in enumerate(scenes) if not sc["title"] or not sc["bullets"]]
    if not todo:
        return
    if use_groq:
        try:
            system = (
                "You create slide text for a tech explainer video. You get numbered narration "
                'passages. Reply with JSON only: {"slides": [{"title": str, "bullets": [str, ...]}]} '
                "with exactly one entry per passage, in order. Titles max 6 words; 2-4 bullets, "
                "each max 10 words, summarizing the passage."
            )
            user = "\n\n".join(f"{n + 1}. {scenes[i]['narration']}" for n, i in enumerate(todo))
            slides = groq_json(system, user).get("slides", [])
            if len(slides) == len(todo):
                for i, sl in zip(todo, slides):
                    scenes[i]["title"] = scenes[i]["title"] or sl.get("title", "")
                    scenes[i]["bullets"] = scenes[i]["bullets"] or sl.get("bullets", [])
        except Exception as e:  # fall back to offline heuristic
            print(f"  ! Groq slide generation failed ({e}); using simple fallback", file=sys.stderr)
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

    img.save(out_png)


# ---------- 4. Assemble (MoviePy 2.x) ----------

def build_video(scenes, voice_model, workdir: Path, out_mp4: Path, on_progress=None, quiet=False) -> float:
    """Render scenes to out_mp4. on_progress(fraction 0-1, message) is optional. Returns duration in seconds."""
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips, vfx

    def report(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(frac, msg)

    workdir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, scene in enumerate(scenes):
        msg = f"Preparing scene {i + 1}/{len(scenes)}: {scene.get('title')}"
        print(msg)
        report(0.3 * i / len(scenes), msg)
        png = workdir / f"slide_{i:02d}.png"
        wav = workdir / f"voice_{i:02d}.wav"

        draw_slide(scene, i, len(scenes), png)
        narration = scene.get("narration", "")
        audio_len = synthesize(narration, wav, voice_model)
        duration = (audio_len + 0.6) if audio_len else estimate_duration(narration)

        clip = ImageClip(str(png)).with_duration(duration)
        if audio_len:
            clip = clip.with_audio(AudioFileClip(str(wav)))
        clip = clip.with_effects([vfx.FadeIn(0.4), vfx.FadeOut(0.4)])
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    report(0.3, "Rendering video...")

    logger = "bar"
    if quiet or on_progress:
        from proglog import ProgressBarLogger

        class _Prog(ProgressBarLogger):
            def bars_callback(self, bar, attr, value, old_value=None):
                if bar == "frame_index" and attr == "index":
                    total = self.bars[bar].get("total") or 1
                    report(0.3 + 0.7 * min(value / total, 1.0), "Rendering video...")

        logger = _Prog() if on_progress else None

    final.write_videofile(str(out_mp4), fps=FPS, codec="libx264", audio_codec="aac", logger=logger)
    duration = float(final.duration)
    print(f"\nDone: {out_mp4}")
    return duration


def main() -> None:
    ap = argparse.ArgumentParser(description="Topic -> explainer video using Groq + MoviePy")
    ap.add_argument("topic", nargs="?", help="Video topic, e.g. 'How does DNS work?'")
    ap.add_argument("--script", help="Use your own script file (.txt, .md or .json) instead of Groq writing it")
    ap.add_argument("--no-groq-slides", action="store_true",
                    help="With --script: don't use Groq to make titles/bullets, use a simple offline fallback")
    ap.add_argument("--scenes", type=int, default=5, help="Number of scenes (default 5)")
    ap.add_argument("--voice", help="Path to a Piper .onnx voice model (optional)")
    ap.add_argument("--out", default="output.mp4", help="Output file name")
    ap.add_argument("--demo", action="store_true", help="Skip Groq, use built-in sample scenes")
    args = ap.parse_args()

    if args.demo:
        scenes = DEMO_SCENES
    elif args.script:
        scenes = parse_script(Path(args.script))
        if not scenes:
            sys.exit("No scenes found in your script file.")
        use_groq = bool(os.environ.get("GROQ_API_KEY")) and not args.no_groq_slides
        fill_missing_slides(scenes, use_groq)
        print(f"Loaded {len(scenes)} scenes from {args.script}")
    else:
        if not args.topic:
            ap.error("Provide a topic, --script FILE, or --demo")
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("Set GROQ_API_KEY first (free key at console.groq.com).")
        print(f"Asking Groq for a script on: {args.topic}")
        scenes = generate_scenes(args.topic, args.scenes)

    workdir = Path("build")
    workdir.mkdir(exist_ok=True)
    (workdir / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")  # edit & reuse if you like

    build_video(scenes, args.voice, workdir, Path(args.out))


if __name__ == "__main__":
    main()
