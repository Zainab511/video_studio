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

import llm
import themes

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads GROQ_API_KEY / GROQ_MODEL from a .env file next to where you run the script
except ImportError:
    pass  # fine if you export the variables in your shell instead

FPS = int(os.getenv("VIDEO_FPS", "15"))               # slides are still images, so low fps is fine; lower = faster
OUT_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1080"))   # output height, e.g. 720 or 480 on weak servers
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "veryfast")  # x264 speed preset: ultrafast (fastest) ... medium (smallest files)
VIDEO_THREADS = int(os.getenv("VIDEO_THREADS", "2"))
VIDEO_THEME = os.getenv("VIDEO_THEME", "auto")                    # auto | mix | midnight | neon | sunrise | ...
VIDEO_ANIMATED = os.getenv("VIDEO_ANIMATED", "1") != "0"          # 0 = themed still slides (fastest)

# Voice and audio quality
PIPER_NOISE_SCALE = os.getenv("PIPER_NOISE_SCALE", "0.5")   # Piper default 0.667. Lower = smoother, less rough/buzzy
PIPER_NOISE_W = os.getenv("PIPER_NOISE_W", "")              # Piper default 0.8. Lower = steadier rhythm
PIPER_LENGTH_SCALE = os.getenv("PIPER_LENGTH_SCALE", "")    # 1.0 = normal pace, 1.1 = 10% slower
AUDIO_CLEANUP = os.getenv("AUDIO_CLEANUP", "light").lower()  # off | light | strong
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "192k")
AUDIO_RATE = 48000

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


def synthesize(text: str, out_wav: Path, voice_model: str | None,
               noise_scale: str | float | None = None, noise_w: str | float | None = None,
               length_scale: str | float | None = None) -> float | None:
    """Return audio duration in seconds, or None if no TTS is available."""
    if not voice_model:
        return None
    piper = shutil.which("piper")
    if not piper:
        print("  ! 'piper' not found on PATH, skipping voiceover", file=sys.stderr)
        return None
    ns = PIPER_NOISE_SCALE if noise_scale is None else noise_scale
    nw = PIPER_NOISE_W if noise_w is None else noise_w
    ls = PIPER_LENGTH_SCALE if length_scale is None else length_scale
    cmd = [piper, "--model", voice_model, "--output_file", str(out_wav)]
    for flag, val in (("--noise-scale", ns), ("--noise-w-scale", nw), ("--length-scale", ls)):
        if str(val).strip():
            cmd += [flag, str(val)]
    r = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("piper failed: " + r.stderr.decode("utf-8", "replace")[-400:])
    return wav_duration(out_wav)


def estimate_duration(text: str) -> float:
    words = len(text.split())
    return max(4.0, words / 2.6)  # ~155 words per minute


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


def audio_filter(level: str) -> str:
    """ffmpeg filter chain, applied once to the whole narration track."""
    level = (level or "light").lower()
    if level == "off":
        return ""
    chain = ["highpass=f=75",      # remove low rumble
             "lowpass=f=10500"]    # cut the hissy top end (the voice has little above this)
    if level == "strong":
        chain.append("afftdn=nr=12:nf=-40")   # spectral noise reduction; can make the voice slightly "watery"
    chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")  # even loudness, YouTube-friendly level
    chain.append(f"aresample={AUDIO_RATE}")
    return ",".join(chain)


def build_narration(prepared, workdir: Path) -> Path:
    """Join all scene voiceovers into ONE continuous, sample-accurate track (no per-scene re-encoding)."""
    parts = []
    for i, (_png, wav, dur) in enumerate(prepared):
        part = workdir / f"narr_{i:02d}.wav"
        if wav:  # resample, then pad with silence up to the scene length
            run_ffmpeg(["-i", str(wav), "-af", f"aresample={AUDIO_RATE},apad", "-t", f"{dur:.3f}",
                        "-ac", "1", "-c:a", "pcm_s16le", str(part)])
        else:
            run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=mono", "-t", f"{dur:.3f}",
                        "-c:a", "pcm_s16le", str(part)])
        parts.append(part)
    listing = workdir / "narr_list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    out = workdir / "narration.wav"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing.name, "-c", "copy", str(out.resolve())], cwd=workdir)
    return out


def render_scene(i: int, n: int, scene: dict, dur: float, th, animated: bool, workdir: Path) -> Path:
    """Render one scene to a silent MP4 clip and return its path."""
    S = OUT_HEIGHT / themes.H
    seg = workdir / f"seg_{i:02d}.mp4"
    art = themes.scene_art(scene, i, n, th, dur)
    is_last = i == n - 1
    common = ["-t", f"{dur:.3f}", "-r", str(FPS), "-c:v", "libx264", "-preset", VIDEO_PRESET,
              "-threads", str(VIDEO_THREADS), "-an"]
    if animated:
        inputs, graph = themes.animated_graph(art, workdir, f"s{i:02d}", S, FPS, dur, is_last)
        run_ffmpeg([*inputs, "-filter_complex", graph, "-map", "[vout]", *common, str(seg)])
    else:
        png = workdir / f"slide_{i:02d}.png"
        themes.composite(art, S).save(png)
        fade = min(0.4, dur / 4)
        vf = f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={dur - fade:.3f}:d={fade:.2f},format=yuv420p"
        run_ffmpeg(["-loop", "1", "-framerate", str(FPS), "-i", str(png), "-vf", vf,
                    *common, "-tune", "stillimage", str(seg)])
    return seg


def build_video(scenes, voice_model, workdir: Path, out_mp4: Path, on_progress=None, quiet=False,
                theme: str | None = None, animated: bool | None = None) -> float:
    """
    Render scenes to out_mp4. on_progress(fraction 0-1, message) is optional. Returns duration in seconds.

    theme: a theme id, 'auto' (match the topic) or 'mix' (a new theme each scene). animated: False renders
    themed still slides, which is much faster. Each scene becomes a short clip; clips are joined without
    re-encoding. Narration is one continuous track, cleaned once and encoded to AAC once.
    """
    theme = theme or VIDEO_THEME
    animated = VIDEO_ANIMATED if animated is None else animated

    def report(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(frac, msg)

    workdir.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(out_mp4)
    n = len(scenes)

    # Phase 1: voiceover audio + scene lengths
    prepared = []  # (None, wav or None, duration)
    for i, scene in enumerate(scenes):
        msg = f"Preparing scene {i + 1}/{n}: {scene.get('title')}"
        print(msg)
        report(0.1 * i / n, msg)
        wav = workdir / f"voice_{i:02d}.wav"
        narration = scene.get("narration", "")
        audio_len = synthesize(narration, wav, voice_model)
        duration = (audio_len + 0.6) if audio_len else estimate_duration(narration)
        prepared.append((None, wav if audio_len else None, duration))

    has_audio = any(w for _, w, _ in prepared)

    # Phase 2: one silent, themed video clip per scene
    segments = []
    for i, (_none, _wav, dur) in enumerate(prepared):
        report(0.1 + 0.78 * i / n, f"Rendering scene {i + 1}/{n}...")
        th = themes.pick_theme(theme, scenes, i)
        segments.append(render_scene(i, n, scenes[i], dur, th, animated, workdir))

    # Phase 3: join the clips (no re-encoding)
    report(0.88, "Joining scenes...")
    listing = workdir / "list.txt"
    listing.write_text("".join(f"file '{seg.name}'\n" for seg in segments), encoding="utf-8")
    video_only = (workdir / "video_all.mp4") if has_audio else out_mp4.resolve()
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
                "-movflags", "+faststart", str(video_only.resolve())], cwd=workdir)

    # Phase 4: narration track -> cleaned, encoded once, added to the video
    if has_audio:
        report(0.93, "Mixing narration...")
        narration_wav = build_narration(prepared, workdir)
        cmd = ["-i", str(video_only), "-i", str(narration_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
        af = audio_filter(AUDIO_CLEANUP)
        if af:
            cmd += ["-af", af]
        cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", str(AUDIO_RATE), "-ac", "2",
                "-shortest", "-movflags", "+faststart", str(out_mp4.resolve())]
        run_ffmpeg(cmd)

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
    ap.add_argument("--theme", default=None, choices=["auto", "mix", *themes.THEME_IDS],
                    help="Visual theme (default: VIDEO_THEME from .env, else auto)")
    ap.add_argument("--static", action="store_true", help="Themed still slides, no animation (fastest render)")
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

    build_video(scenes, args.voice, workdir, Path(args.out), theme=args.theme, animated=not args.static)


if __name__ == "__main__":
    main()