"""
Visual themes for Video Studio.

Each scene becomes a "SceneArt": an animated-capable background plus layered elements
(decor, title, bullets, badge, progress bar). Two ways to turn it into video:

  composite()      -> one still image (fast "static" mode, still themed)
  animated_graph() -> an ffmpeg filter graph: slow background drift, floating shapes,
                      text sliding/fading in, blinking cursor, scrolling waves, and a
                      wipe transition between scenes.

Everything is drawn with Pillow at 1080p "base" coordinates and scaled to the output size.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H = 1920, 1080
BG_MARGIN = 0.12  # extra background around the frame so it can drift without showing edges


# ---------------------------------------------------------------- themes

@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    stops: tuple
    gdir: str = "v"
    text: str = "#ffffff"
    sub: str = "#dbe4f0"
    accent: str = "#38bdf8"
    accent2: str = "#a78bfa"
    curtain: str = "#38bdf8"
    font: str = "sans"
    bullet: str = "dot"      # dot | card | square | prompt
    decor: str = "orbs"
    glow: str | None = None  # optional glow colour behind the title
    dark: bool = True


THEME_LIST = [
    Theme("midnight", "Midnight", ("#0b1026", "#1a1f4d"), "d", accent="#4cc9f0", accent2="#7b5cff",
          curtain="#4cc9f0", sub="#cfd8ee", decor="stars"),
    Theme("neon", "Neon Grid", ("#07030f", "#1b0b3a", "#07030f"), "v", accent="#00f0ff", accent2="#ff2bd6",
          curtain="#ff2bd6", sub="#e8d7ff", decor="grid", glow="#00f0ff"),
    Theme("sunrise", "Sunrise", ("#ff9a3c", "#ff5e7e", "#4a2a8a"), "v", accent="#ffe066", accent2="#ffffff",
          curtain="#ff5e7e", sub="#fff1e6", bullet="card", decor="sun"),
    Theme("forest", "Forest", ("#062e24", "#0f5c4d"), "d", text="#f2fff6", sub="#d2f5df", accent="#b6f26a",
          accent2="#4ade80", curtain="#4ade80", decor="leaves"),
    Theme("blueprint", "Blueprint", ("#0c4a8f", "#082b57"), "d", accent="#ffd23f", accent2="#9cc9ff",
          curtain="#ffd23f", sub="#cfe3ff", bullet="square", decor="blueprint"),
    Theme("terminal", "Terminal", ("#04080a", "#0a1a12"), "v", text="#7cffb2", sub="#a8e6c0", accent="#39ff88",
          accent2="#1fbf6b", curtain="#39ff88", font="mono", bullet="prompt", decor="rain", glow="#39ff88"),
    Theme("clean", "Clean Light", ("#f7f9ff", "#e3eaf8"), "d", text="#14213d", sub="#33415c", accent="#3b82f6",
          accent2="#ec4899", curtain="#3b82f6", bullet="card", decor="blobs", dark=False),
    Theme("ocean", "Ocean", ("#05264a", "#0b5d73", "#0f8f8a"), "v", accent="#7ee7ff", accent2="#c4f5ff",
          curtain="#0ea5b7", sub="#d7f3f5", bullet="card", decor="ocean"),
]
THEMES = {t.id: t for t in THEME_LIST}
THEME_IDS = [t.id for t in THEME_LIST]

KEYWORDS = {
    "terminal": "code coding program programming python javascript git github api server linux terminal command "
                "software debug developer database sql docker script bash cli".split(),
    "neon": "ai machine learning neural gpt model crypto blockchain cyber hacking security gaming game future "
            "robot automation".split(),
    "ocean": "water ocean sea cloud network internet wifi wave flow stream traffic dns".split(),
    "forest": "nature environment plant climate green health biology animal tree garden energy sustainable".split(),
    "sunrise": "motivation startup business productivity success morning habit growth career money finance "
               "marketing".split(),
    "blueprint": "engineering design architecture system hardware electronics math physics science chemistry "
                 "circuit mechanical build".split(),
    "clean": "education tutorial guide basics beginner learn lesson explain introduction overview".split(),
    "midnight": "space night astronomy star universe history mystery".split(),
}


def pick_theme(setting: str | None, scenes: list[dict], index: int) -> Theme:
    """setting: a theme id, 'auto' (match the topic), or 'mix' (a new theme each scene)."""
    setting = (setting or "auto").lower()
    if setting in THEMES:
        return THEMES[setting]
    text = " ".join(f"{s.get('title', '')} {s.get('narration', '')}" for s in scenes).lower()
    seed = sum(ord(c) for c in (scenes[0].get("title", "x") if scenes else "x"))
    if setting == "mix":
        return THEME_LIST[(seed + index) % len(THEME_LIST)]
    words = [w.strip(".,!?;:()\"'") for w in text.split()]
    scores = {tid: sum(words.count(k) for k in kws) for tid, kws in KEYWORDS.items()}
    best = max(scores.values()) if scores else 0
    if best > 0:
        return THEMES[max(scores, key=scores.get)]
    return THEME_LIST[seed % len(THEME_LIST)]


# ---------------------------------------------------------------- drawing helpers

def rgb(c: str) -> tuple:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


_FONT_FILES = {
    "sans": ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf",
             "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "sans_bold": ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "mono": ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "C:/Windows/Fonts/consola.ttf", "/System/Library/Fonts/Menlo.ttc"],
    "mono_bold": ["DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                  "C:/Windows/Fonts/consolab.ttf", "/System/Library/Fonts/Menlo.ttc"],
}


@lru_cache(maxsize=64)
def font(role: str, size: int):
    for f in _FONT_FILES.get(role, _FONT_FILES["sans"]):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def gradient(size: tuple, stops: tuple, direction: str = "v") -> Image.Image:
    w, h = size
    base = Image.linear_gradient("L")  # black at top -> white at bottom
    if direction == "h":
        g = base.rotate(90, expand=True).resize((w, h), Image.BICUBIC)
    elif direction == "d":
        m = w + h
        g = base.resize((m, m), Image.BICUBIC).rotate(45, expand=True, resample=Image.BICUBIC)
        gx, gy = (g.width - w) // 2, (g.height - h) // 2
        g = g.crop((gx, gy, gx + w, gy + h))
    else:
        g = base.resize((w, h), Image.BICUBIC)
    mid = rgb(stops[1]) if len(stops) == 3 else None
    return ImageOps.colorize(g, black=rgb(stops[0]), white=rgb(stops[-1]), mid=mid).convert("RGBA")


def orb(d: int, color: str, alpha: float = 0.5) -> Image.Image:
    """Soft glowing circle."""
    # Pillow's radial gradient only reaches ~70% at the edge midpoints, so rescale it to hit 100% at the edge
    g = Image.radial_gradient("L").point(lambda v: min(255, int(v * 1.4143)))
    m = ImageOps.invert(g).resize((d, d), Image.BICUBIC)
    m = m.point(lambda v: int(255 * ((v / 255) ** 1.5) * alpha))
    img = Image.new("RGBA", (d, d), rgb(color) + (0,))
    img.putalpha(m)
    return img


def ring(d: int, color: str, width: int, alpha: int = 200, dashes: int = 0) -> Image.Image:
    s = 3
    img = Image.new("RGBA", (d * s, d * s), rgb(color) + (0,))
    dr = ImageDraw.Draw(img)
    box = [width * s, width * s, d * s - width * s, d * s - width * s]
    col = rgb(color) + (alpha,)
    if dashes:
        step = 360 / dashes
        for k in range(dashes):
            dr.arc(box, k * step, k * step + step * 0.55, fill=col, width=width * s)
    else:
        dr.ellipse(box, outline=col, width=width * s)
    return img.resize((d, d), Image.LANCZOS)


def wrap_px(text: str, fnt, maxw: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if fnt.getlength(test) <= maxw:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def text_img(lines, fnt, fill, shadow: bool = False, glow: str | None = None, alpha: int = 255, pad: int = 30):
    """Transparent image of text. Returns (image, text_width, text_height)."""
    line_h = int(fnt.size * 1.25)
    tw = max(int(fnt.getlength(ln)) for ln in lines)
    th = line_h * len(lines)
    size = (tw + 2 * pad, th + 2 * pad)
    col = rgb(fill)

    def layer(color, a=255):
        img = Image.new("RGBA", size, color + (0,))
        d = ImageDraw.Draw(img)
        for k, ln in enumerate(lines):
            d.text((pad, pad + k * line_h), ln, font=fnt, fill=color + (a,))
        return img

    out = Image.new("RGBA", size, col + (0,))
    if glow:
        g = layer(rgb(glow), 255).filter(ImageFilter.GaussianBlur(16))
        g.putalpha(g.getchannel("A").point(lambda v: int(v * 0.8)))
        out = Image.alpha_composite(out, g)
    if shadow:
        sh = layer((0, 0, 0), 150).filter(ImageFilter.GaussianBlur(5))
        out = Image.alpha_composite(out, ImageChops_offset(sh, 3, 5))
    out = Image.alpha_composite(out, layer(col, alpha))
    return out, tw, th


def ImageChops_offset(img: Image.Image, dx: int, dy: int) -> Image.Image:
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (dx, dy))
    return out


def card(w: int, h: int, fill: tuple, outline: tuple | None = None, shadow: bool = False, radius: int = 28):
    pad = 34 if shadow else 4
    s = 2
    img = Image.new("RGBA", ((w + 2 * pad) * s, (h + 2 * pad) * s), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([pad * s, (pad + 10) * s, (pad + w) * s, (pad + 10 + h) * s],
                                             radius=radius * s, fill=(20, 30, 70, 70))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(16)))
    ImageDraw.Draw(img).rounded_rectangle([pad * s, pad * s, (pad + w) * s, (pad + h) * s], radius=radius * s,
                                          fill=fill, outline=outline, width=2 * s if outline else 0)
    return img.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS), pad


# ---------------------------------------------------------------- elements and art

@dataclass
class El:
    img: Image.Image
    x: float
    y: float
    slide: tuple | None = None    # (dx, dy, start, duration): starts offset by dx,dy and eases into place
    fade: tuple | None = None     # (start, duration)
    drift: tuple | None = None    # (ax, ay, period_seconds, phase)
    scroll: tuple | None = None   # ("x"|"y", speed px/s, period px): endless scrolling
    rise: tuple | None = None     # (speed px/s, travel px, offset px): rises and wraps
    blink: tuple | None = None    # (period, duty, phase)
    spin: float | None = None     # seconds per full rotation


@dataclass
class SceneArt:
    bg: Image.Image
    pan: tuple
    els: list
    curtain: str


def _rnd(theme: Theme, i: int) -> random.Random:
    return random.Random(f"{theme.id}-{i}")


def make_background(th: Theme, rnd: random.Random):
    """Returns (background image, pan spec)."""
    d = th.decor
    if d in ("grid", "blueprint"):
        period = 120 if d == "grid" else 200
        bg = gradient((W + period, H + period), th.stops, th.gdir)
        dr = ImageDraw.Draw(bg, "RGBA")
        if d == "grid":
            col = rgb(th.accent2) + (46,)
            for x in range(0, W + period, period):
                dr.line([(x, 0), (x, H + period)], fill=col, width=2)
            for y in range(0, H + period, period):
                dr.line([(0, y), (W + period, y)], fill=col, width=2)
            return bg, ("scroll", 18, 30, period, period)
        fine, thick = rgb("#ffffff") + (22,), rgb("#ffffff") + (52,)
        for x in range(0, W + period, 40):
            dr.line([(x, 0), (x, H + period)], fill=thick if x % 200 == 0 else fine, width=2 if x % 200 == 0 else 1)
        for y in range(0, H + period, 40):
            dr.line([(0, y), (W + period, y)], fill=thick if y % 200 == 0 else fine, width=2 if y % 200 == 0 else 1)
        return bg, ("scroll", 10, 6, period, period)
    mx, my = int(W * BG_MARGIN), int(H * BG_MARGIN)
    bg = gradient((W + mx, H + my), th.stops, th.gdir)
    if d == "rain":  # faint scanlines
        dr = ImageDraw.Draw(bg, "RGBA")
        for y in range(0, H + my, 4):
            dr.line([(0, y), (W + mx, y)], fill=(0, 0, 0, 38), width=1)
        return bg, ("none",)
    return bg, ("sin", 90, 55, 22.0)


def _stars(rnd, count, color, alpha=(120, 235)):
    img = Image.new("RGBA", (W, H), rgb(color) + (0,))
    d = ImageDraw.Draw(img)
    for _ in range(count):
        r = rnd.choice([1.5, 2, 2, 2.5, 3])
        x, y = rnd.uniform(0, W), rnd.uniform(0, H)
        d.ellipse([x - r, y - r, x + r, y + r], fill=rgb(color) + (rnd.randint(*alpha),))
    return img


def make_decor(th: Theme, i: int, rnd: random.Random) -> list:
    els: list = []
    ph = lambda: rnd.uniform(0, 6.28)
    d = th.decor

    def floaty(img, x, y, ax=90, ay=60, p=12.0):
        els.append(El(img, x, y, drift=(ax * rnd.uniform(0.7, 1.3), ay * rnd.uniform(0.7, 1.3),
                                       p * rnd.uniform(0.8, 1.3), ph())))

    if d == "stars":
        floaty(orb(1000, th.accent2, 0.32), rnd.uniform(-250, 600), rnd.uniform(-300, 300), 120, 80, 15)
        floaty(orb(800, th.accent, 0.26), rnd.uniform(900, 1500), rnd.uniform(300, 700), 110, 90, 13)
        els.append(El(_stars(rnd, 70, "#ffffff"), 0, 0, blink=(3.2, 0.62, ph())))
        els.append(El(_stars(rnd, 55, th.accent), 0, 0, blink=(4.4, 0.58, ph())))
    elif d == "grid":
        floaty(orb(900, th.accent2, 0.30), rnd.uniform(-200, 300), rnd.uniform(300, 600), 100, 60, 14)
        floaty(orb(800, th.accent, 0.24), rnd.uniform(1100, 1500), rnd.uniform(-200, 100), 90, 70, 12)
        bar = Image.new("RGBA", (900, 6), rgb(th.accent) + (255,))
        els.append(El(bar.filter(ImageFilter.GaussianBlur(1)), 110, 70, fade=(0.2, 0.8)))
    elif d == "sun":
        floaty(orb(1100, "#ffd166", 0.95), W - 760, H - 640, 40, 50, 18)
        floaty(orb(560, "#ffffff", 0.22), rnd.uniform(0, 500), rnd.uniform(500, 800), 70, 60, 11)
        floaty(orb(420, "#ffb3c7", 0.30), rnd.uniform(1000, 1500), rnd.uniform(-80, 150), 60, 50, 10)
    elif d == "leaves":
        floaty(orb(900, th.accent2, 0.22), rnd.uniform(1000, 1400), rnd.uniform(-100, 300), 90, 70, 14)
        for _ in range(9):
            s = 3
            lw, lh = rnd.randint(110, 210), rnd.randint(50, 90)
            leaf = Image.new("RGBA", (lw * s, lh * s), rgb(th.accent) + (0,))
            ImageDraw.Draw(leaf).ellipse([0, 0, lw * s - 1, lh * s - 1],
                                         fill=rgb(rnd.choice([th.accent, th.accent2, "#2dd4bf"])) + (rnd.randint(55, 105),))
            leaf = leaf.rotate(rnd.uniform(0, 180), expand=True, resample=Image.BICUBIC)
            leaf = leaf.resize((leaf.width // s, leaf.height // s), Image.LANCZOS)
            floaty(leaf, rnd.uniform(0, W - 200), rnd.uniform(0, H - 200), 80, 110, 9)
    elif d == "blueprint":
        els.append(El(ring(520, th.accent2, 3, 150, dashes=36), W - 640, 130, spin=40.0, fade=(0.3, 0.8)))
        els.append(El(ring(340, th.accent, 3, 190), W - 550, 220, fade=(0.5, 0.8)))
        for cx, cy in ((60, 40), (W - 120, 40), (60, H - 120), (W - 120, H - 120)):
            m = Image.new("RGBA", (60, 60), rgb(th.accent) + (0,))
            md = ImageDraw.Draw(m)
            md.line([(0, 30), (60, 30)], fill=rgb(th.accent) + (200,), width=3)
            md.line([(30, 0), (30, 60)], fill=rgb(th.accent) + (200,), width=3)
            els.append(El(m, cx, cy, fade=(0.2, 0.6)))
    elif d == "rain":
        glyphs = "01{}[]<>/\\$#=+*;"
        fnt = font("mono", 30)
        for k in range(7):
            x = rnd.randint(40, W - 80)
            rows = H // 36
            col = Image.new("RGBA", (50, H * 2), rgb(th.accent) + (0,))
            cd = ImageDraw.Draw(col)
            strip = [rnd.choice(glyphs) for _ in range(rows)]
            for rep in range(2):  # same strip twice so it loops seamlessly
                for r, ch in enumerate(strip):
                    cd.text((10, rep * H + r * 36), ch, font=fnt, fill=rgb(th.accent) + (rnd.randint(40, 110),))
            els.append(El(col, x, 0, scroll=("y", rnd.uniform(40, 110), H)))
    elif d == "blobs":
        floaty(orb(1000, th.accent, 0.50), rnd.uniform(-300, 200), rnd.uniform(300, 600), 110, 70, 15)
        floaty(orb(850, th.accent2, 0.38), rnd.uniform(1100, 1500), rnd.uniform(-250, 100), 100, 80, 13)
        floaty(orb(700, "#34d399", 0.34), rnd.uniform(800, 1300), rnd.uniform(500, 800), 90, 60, 12)
    elif d == "ocean":
        for k, (col, a, v, amp, ph0, hh) in enumerate(((th.accent2, 38, 90, 34, 0.0, 300),
                                                      (th.accent, 52, -140, 26, 1.7, 240))):
            wv = Image.new("RGBA", (2 * W, hh), rgb(col) + (0,))
            wd = ImageDraw.Draw(wv)
            pts = [(x, 60 + amp * math.sin(2 * math.pi * 2 * x / W + ph0)) for x in range(0, 2 * W + 1, 8)]
            wd.polygon(pts + [(2 * W, hh), (0, hh)], fill=rgb(col) + (a,))
            els.append(El(wv, 0, H - hh + 20 - k * 20, scroll=("x", v, W)))
        for _ in range(7):
            dd = rnd.randint(26, 70)
            els.append(El(ring(dd, "#ffffff", 3, 130), rnd.uniform(100, W - 100), H + 100,
                          rise=(rnd.uniform(45, 110), H + 260, rnd.uniform(0, 900)),
                          drift=(rnd.uniform(15, 35), 0, rnd.uniform(3, 6), ph())))
        floaty(orb(900, th.accent, 0.20), rnd.uniform(-200, 300), rnd.uniform(-250, 100), 100, 60, 14)
    return els


def foreground(scene: dict, i: int, n: int, th: Theme, dur: float) -> list:
    els: list = []
    title = (scene.get("title") or f"Scene {i + 1}").strip()
    if th.id == "terminal" and not title.startswith(">"):
        title = "> " + title
    tfont = None
    for size in (96, 84, 72, 62):
        tfont = font(f"{th.font}_bold", size)
        tlines = wrap_px(title, tfont, 1560)
        if len(tlines) <= 2:
            break
    tlines = tlines[:2]
    timg, tw, thh = text_img(tlines, tfont, th.text, shadow=th.dark, glow=th.glow)
    pad, tx, ty = 30, 170, 120
    line_h = int(tfont.size * 1.25)

    # accent bar
    bar_h = max(line_h * len(tlines) - 10, 90)
    bar = Image.new("RGBA", (14, bar_h), rgb(th.accent) + (255,))
    els.append(El(bar, 110, ty + 8, slide=(0, -140, 0.2, 0.6), fade=(0.2, 0.5)))
    els.append(El(timg, tx - pad, ty - pad, slide=(-150, 0, 0.35, 0.75), fade=(0.35, 0.55)))

    if th.id == "terminal":  # blinking block cursor after the title
        cur_w, cur_h = int(tfont.size * 0.5), int(tfont.size * 0.95)
        cur = Image.new("RGBA", (cur_w, cur_h), rgb(th.accent) + (235,))
        last_w = int(tfont.getlength(tlines[-1]))
        els.append(El(cur, tx + last_w + 18, ty + (len(tlines) - 1) * line_h + int(tfont.size * 0.12),
                      fade=(1.0, 0.1), blink=(1.0, 0.55, 0.0)))

    # bullets
    bullets = [b for b in scene.get("bullets", []) if str(b).strip()][:4]
    y0 = max(380, ty + line_h * len(tlines) + 70)
    avail = 985 - y0
    built, gap = [], 30
    for size in (54, 50, 46, 42, 38):
        built = [_bullet(str(b), th, size) for b in bullets]
        total = sum(img.height for img in built) + gap * max(len(built) - 1, 0)
        if total <= avail:
            break
    n_b = max(len(built), 1)
    step = min(2.2, max(0.4, (dur * 0.6 - 0.9) / n_b))
    y = y0
    for k, img in enumerate(built):
        t0 = min(0.9 + k * step, max(dur - 1.2, 0.5))
        els.append(El(img, tx - 70, y, slide=(170, 0, t0, 0.65), fade=(t0, 0.5)))
        y += img.height + gap

    # badge + progress
    bf = font("mono", 34)
    label = f"{i + 1:02d} / {n:02d}"
    bimg, bw, _ = text_img([label], bf, th.sub, alpha=190, pad=10)
    els.append(El(bimg, W - 110 - bw - 10, 62, fade=(0.6, 0.5)))
    track_w = W - 220
    track = Image.new("RGBA", (track_w, 8), (255, 255, 255, 46 if th.dark else 0) if th.dark else (20, 33, 61, 40))
    fill = Image.new("RGBA", (max(int(track_w * (i + 1) / n), 8), 8), rgb(th.accent) + (255,))
    els.append(El(track, 110, H - 46))
    els.append(El(fill, 110, H - 46, fade=(0.5, 0.7)))
    return els


def _bullet(text: str, th: Theme, size: int) -> Image.Image:
    fnt = font(th.font, size)
    lines = wrap_px(text, fnt, 1330)[:2]
    timg, tw, thh = text_img(lines, fnt, th.sub if th.id != "clean" else th.text, shadow=th.dark, pad=8)
    ih = timg.height
    icon_w = 78
    if th.bullet == "card":
        fill = (255, 255, 255, 235) if not th.dark else (255, 255, 255, 30)
        outline = (255, 255, 255, 60) if th.dark else (59, 130, 246, 70)
        cw, ch = tw + icon_w + 70, max(thh + 44, 96)
        base, pad = card(cw, ch, fill, outline, shadow=not th.dark)
        strip = Image.new("RGBA", (8, ch - 36), rgb(th.accent) + (255,))
        base.paste(strip, (pad + 22, pad + 18), strip)
        base.paste(timg, (pad + icon_w - 18 + 6, pad + (ch - ih) // 2), timg)
        return base
    canvas = Image.new("RGBA", (tw + icon_w + 70, ih), (0, 0, 0, 0))
    canvas.paste(timg, (icon_w, 0), timg)
    cy = ih // 2 - 4
    s = 4
    if th.bullet == "square":
        icon = Image.new("RGBA", (30 * s, 30 * s), (0, 0, 0, 0))
        ImageDraw.Draw(icon).rectangle([2 * s, 2 * s, 28 * s, 28 * s], outline=rgb(th.accent) + (255,), width=3 * s)
    elif th.bullet == "prompt":
        icon = None
        pf = font("mono_bold", size)
        pimg, _, _ = text_img(["$"], pf, th.accent, pad=8)
        canvas.paste(pimg, (6, 0), pimg)
    else:
        icon = Image.new("RGBA", (30 * s, 30 * s), (0, 0, 0, 0))
        ImageDraw.Draw(icon).ellipse([3 * s, 3 * s, 27 * s, 27 * s], fill=rgb(th.accent) + (255,))
    if icon is not None:
        icon = icon.resize((30, 30), Image.LANCZOS)
        canvas.paste(icon, (22, cy - 15), icon)
    return canvas


def scene_art(scene: dict, i: int, n: int, th: Theme, dur: float) -> SceneArt:
    rnd = _rnd(th, i)
    bg, pan = make_background(th, rnd)
    els = make_decor(th, i, rnd)
    wm = font(f"{th.font}_bold", 430)
    mark, _, _ = text_img([f"{i + 1:02d}"], wm, th.accent, alpha=20 if th.dark else 26, pad=10)
    els.append(El(mark, W - 760, H - 600, drift=(18, 14, 16.0, 0.5)))
    els += foreground(scene, i, n, th, dur)
    return SceneArt(bg=bg, pan=pan, els=els, curtain=th.curtain)


# ---------------------------------------------------------------- static image

def composite(art: SceneArt, S: float = 1.0) -> Image.Image:
    bw, bh = art.bg.size
    cx, cy = (0, 0) if art.pan[0] == "scroll" else ((bw - W) // 2, (bh - H) // 2)
    frame = art.bg.crop((cx, cy, cx + W, cy + H)).convert("RGBA")
    for el in art.els:
        frame.paste(el.img, (int(el.x), int(el.y)), el.img)
    frame = frame.convert("RGB")
    if S != 1.0:
        frame = frame.resize(out_size(S), Image.LANCZOS)
    return frame


def out_size(S: float) -> tuple:
    return (round(W * S) // 2 * 2, round(H * S) // 2 * 2)


# ---------------------------------------------------------------- animated ffmpeg graph

def _f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


def _scaled(img: Image.Image, S: float) -> Image.Image:
    if S == 1.0:
        return img
    return img.resize((max(1, round(img.width * S)), max(1, round(img.height * S))), Image.LANCZOS)


def animated_graph(art: SceneArt, workdir: Path, tag: str, S: float, fps: int, dur: float, is_last: bool):
    """Write PNG assets and return (ffmpeg input args, filter_complex string). Output label is [vout]."""
    ow, oh = out_size(S)
    inputs: list[str] = []
    chains: list[str] = []

    idx_counter = [0]

    def still() -> str:
        """Turn a single decoded image into a stream of frames lasting the whole scene (decoded only once)."""
        return f"loop=loop=-1:size=1:start=0,setpts=N/({fps}*TB),trim=duration={_f(dur + 0.5)}"

    def push(img: Image.Image, name: str) -> int:
        p = workdir / f"{tag}_{name}.png"
        img.save(p)
        inputs.extend(["-i", str(p)])
        i = idx_counter[0]
        idx_counter[0] += 1
        return i

    # background + pan
    bg = art.bg.convert("RGB")
    bg = _scaled(bg, S)
    b = push(bg, "bg")
    bwid, bhei = bg.size
    pan = art.pan
    if pan[0] == "scroll":
        _, vx, vy, px, py = pan
        cx = f"mod(t*{_f(vx * S)},{_f(px * S)})"
        cy = f"mod(t*{_f(vy * S)},{_f(py * S)})"
    elif pan[0] == "sin":
        _, ax, ay, per = pan
        cx = f"{_f((bwid - ow) / 2)}+{_f(ax * S)}*sin(2*PI*t/{_f(per)})"
        cy = f"{_f((bhei - oh) / 2)}+{_f(ay * S)}*sin(2*PI*t/{_f(per * 1.3)}+1)"
    else:
        cx, cy = _f((bwid - ow) / 2), _f((bhei - oh) / 2)
    chains.append(f"[{b}:v]{still()},crop=w={ow}:h={oh}:x='{cx}':y='{cy}'[base0]")
    cur = "base0"

    for k, el in enumerate(art.els):
        img = _scaled(el.img, S)
        idx = push(img, f"e{k:02d}")
        src = f"{idx}:v"
        pre = [still()]
        if el.fade or el.spin:
            pre.append("format=rgba")
        if el.fade:
            pre.append(f"fade=t=in:st={_f(el.fade[0])}:d={_f(el.fade[1])}:alpha=1")
        if el.spin:
            pre.append(f"rotate=a='2*PI*t/{_f(el.spin)}':ow=iw:oh=ih:c=none")
        chains.append(f"[{src}]{','.join(pre)}[el{k}]")
        src = f"el{k}"
        x, y = _f(el.x * S), _f(el.y * S)
        xs, ys = x, y
        if el.slide:
            dx, dy, t0, d = el.slide
            e = f"pow(1-clip((t-{_f(t0)})/{_f(d)},0,1),3)"
            if dx:
                xs += f"+({_f(dx * S)})*{e}"
            if dy:
                ys += f"+({_f(dy * S)})*{e}"
        if el.drift:
            ax, ay, per, phase = el.drift
            xs += f"+({_f(ax * S)})*sin(2*PI*t/{_f(per)}+{_f(phase)})"
            ys += f"+({_f(ay * S)})*sin(2*PI*t/{_f(per * 0.83)}+{_f(phase * 1.7)})"
        if el.scroll:
            axis, v, period = el.scroll
            if axis == "x":
                xs = f"{x}-mod(t*{_f(v * S)},{_f(period * S)})" if v >= 0 else \
                    f"{x}-{_f(period * S)}+mod(t*{_f(-v * S)},{_f(period * S)})"
            else:
                ys = f"{y}-mod(t*{_f(v * S)},{_f(period * S)})"
        if el.rise:
            v, travel, off = el.rise
            ys = f"{y}-mod(t*{_f(v * S)}+{_f(off * S)},{_f(travel * S)})"
        opts = f"x='{xs}':y='{ys}'"
        if el.blink:
            per, duty, phase = el.blink
            opts += f":enable='lt(mod(t+{_f(phase)},{_f(per)}),{_f(per * duty)})'"
        nxt = f"base{k + 1}"
        chains.append(f"[{cur}][{src}]overlay={opts}[{nxt}]")
        cur = nxt

    # wipe transition: curtain slides out at the start; slides in at the end (except the last scene)
    c = idx_counter[0]
    inputs.extend(["-f", "lavfi", "-t", _f(dur + 0.5), "-i", f"color=c=0x{art.curtain.lstrip('#')}:s={ow}x{oh}:r={fps}"])
    d = 0.5
    start = f"{ow}*pow(clip(t/{d},0,1),2)"
    if is_last:
        cx_expr = start
    else:
        te = max(dur - d, d + 0.1)
        cx_expr = f"if(lt(t,{_f(te)}),{start},-{ow}*pow(1-clip((t-{_f(te)})/{d},0,1),2))"
    chains.append(f"[{cur}][{c}:v]overlay=x='{cx_expr}':y=0[cw]")
    tail = "[cw]"
    if is_last:
        chains.append(f"{tail}fade=t=out:st={_f(max(dur - 0.6, 0.5))}:d=0.6,format=yuv420p[vout]")
    else:
        chains.append(f"{tail}format=yuv420p[vout]")
    return inputs, ";".join(chains)


# ---------------------------------------------------------------- previews

_SAMPLE = {"title": "Your video title", "bullets": ["Clear key points", "Smooth animated entrances",
                                                    "Matching colours and motion"],
           "narration": "Sample."}
_preview_cache: dict = {}


def preview_png(theme_id: str) -> bytes:
    if theme_id not in THEMES:
        raise KeyError(theme_id)
    if theme_id not in _preview_cache:
        th = THEMES[theme_id]
        img = composite(scene_art(_SAMPLE, 1, 5, th, 12.0), 1 / 3)
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        _preview_cache[theme_id] = buf.getvalue()
    return _preview_cache[theme_id]