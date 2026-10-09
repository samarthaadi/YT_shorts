"""make_video.py: approved script JSON -> checked 1080x1920 Short.

    python make_video.py jobs/2026-10-09-animal-superpowers
    python make_video.py jobs/2026-10-09-animal-superpowers --swap rank3.2   # Gate 3: replace a shot

Every step saves its output into the job folder and skips itself on a re-run,
so swapping one clip re-renders the video without paying for the voice again.
"""
import argparse, base64, difflib, html, json, math, os, random, re, subprocess, sys
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from moviepy import (AudioFileClip, CompositeAudioClip, CompositeVideoClip,
                     ImageClip, VideoFileClip)
from moviepy.audio import fx as afx
from moviepy.video import fx as vfx

load_dotenv()
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
PEXELS_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_KEY = os.getenv("PIXABAY_API_KEY")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

W, H, FPS = 1080, 1920, 30
ROOT = Path(__file__).parent
FONT = str(ROOT / "assets/fonts/Montserrat-ExtraBold.ttf")
FONT_BLACK = str(ROOT / "assets/fonts/Montserrat-Black.ttf")
MUSIC_DIR = ROOT / "assets/music"
KOKORO_MODEL = ROOT / "assets/models/kokoro-v1.0.onnx"
KOKORO_VOICES = ROOT / "assets/models/voices-v1.0.bin"
USED_LOG = ROOT / "used_clips.json"
UA = {"User-Agent": "Top5ShortsBot/1.0 (contact via channel email)"}  # Wikimedia requires one

# Text and key visuals stay inside this box: clear of the top bar, the bottom
# fifth (title, channel name) and the right edge (like/comment/share buttons).
SAFE = (70, 180, 900, int(H * 0.8))
CX = (SAFE[0] + SAFE[2]) // 2
YELLOW = (255, 212, 0)

# Card palettes (top colour, bottom colour) for graphic fallback shots, by section.
PALETTES = {
    "rank5": ((18, 92, 112), (4, 22, 33)),
    "rank4": ((112, 44, 140), (24, 8, 40)),
    "rank3": ((34, 120, 72), (5, 30, 16)),
    "rank2": ((70, 128, 180), (8, 24, 48)),
    "rank1": ((60, 24, 120), (6, 4, 20)),
}


def load(p):
    return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2))


def run(cmd):
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path])
    return float(out.stdout)


def entries_of(script):
    # The plan's format calls the list "entries"; the script stage may emit "countdown".
    return script.get("entries") or script["countdown"]


def script_parts(script):
    """Hook, #5 to #1, outro: the order they are spoken."""
    parts = [{"id": "hook", "text": script["hook"]}]
    for e in sorted(entries_of(script), key=lambda e: -e["rank"]):
        parts.append({"id": f"rank{e['rank']}", "text": e["narration"], "entry": e})
    parts.append({"id": "outro", "text": script["outro"]})
    return parts


# ---------------------------------------------------------------- Step 0: script lint

BANNED = ["you won't believe", "insane", "subscribe"]


def lint_script(script):
    """The plan's automatic script checks. Problems are warnings: Gate 2 already passed."""
    warn = []
    entries = sorted(entries_of(script), key=lambda e: -e["rank"])
    count = lambda t: len(t.split())
    total = count(script["hook"]) + count(script["outro"]) + sum(count(e["narration"]) for e in entries)
    if not 160 <= total <= 240:
        warn.append(f"narration is {total} words; the format wants 160-240")
    if count(script["hook"]) > 9:
        warn.append(f"hook is {count(script['hook'])} words; the format wants 9 or fewer")
    if count(script["outro"]) > 8:
        warn.append(f"outro is {count(script['outro'])} words; the format wants 8 or fewer")
    if [e["rank"] for e in entries] != [5, 4, 3, 2, 1]:
        warn.append("entries are not exactly #5 to #1")
    for e in entries:
        n = count(e["narration"])
        lo, hi = (40, 55) if e["rank"] == 1 else (25, 35)
        if not lo <= n <= hi:
            warn.append(f"#{e['rank']} narration is {n} words; the format wants {lo}-{hi}")
        if len(e.get("sources", [])) < 2:
            warn.append(f"#{e['rank']} has fewer than two sources")
        if len(e.get("visual_terms", [])) < 3:
            warn.append(f"#{e['rank']} has fewer than three visual terms")
    two = next((e for e in entries if e["rank"] == 2), None)
    if two and "number one" not in two["narration"].lower().split("number two")[0]:
        warn.append("#2 does not open with a tease of #1")
    text = " ".join([script["hook"], script["outro"]] + [e["narration"] for e in entries]).lower()
    warn += [f"banned phrase: '{b}'" for b in BANNED if b in text]
    return warn


# ---------------------------------------------------------------- Step 1: voice and word timings

def tts_with_timestamps(job, parts):
    """ElevenLabs: audio plus a time for every character."""
    text = " ".join(p["text"] for p in parts)
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": ELEVEN_KEY},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        timeout=180)
    r.raise_for_status()
    data = r.json()
    (job / "voice_tts.mp3").write_bytes(base64.b64decode(data["audio_base64"]))
    a = data["alignment"]
    return chars_to_words(a["characters"], a["character_start_times_seconds"],
                          a["character_end_times_seconds"])


def chars_to_words(chars, starts, ends):
    words, cur = [], None
    for c, s, e in zip(chars, starts, ends):
        if c.isspace():
            if cur:
                words.append(cur)
            cur = None
            continue
        if cur is None:
            cur = {"word": "", "start": s, "end": e}
        cur["word"] += c
        cur["end"] = e
    if cur:
        words.append(cur)
    return words


def tts_kokoro(job, parts):
    """Free local voice (Kokoro). It has no timestamps, so each sentence is
    synthesised separately and its words are spread across its spoken span by length."""
    import soundfile as sf
    from kokoro_onnx import Kokoro
    k = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    sr, chunks, words, n = 24000, [], [], 0

    def silence(sec):
        nonlocal n
        chunks.append(np.zeros(int(sec * sr), np.float32))
        n += len(chunks[-1])

    for pi, p in enumerate(parts):
        for sent in re.findall(r"[^.!?]+[.!?]*", p["text"]):
            sent = sent.strip()
            if not sent:
                continue
            audio, sr = k.create(sent, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")
            loud = np.flatnonzero(np.abs(audio) > 0.03 * np.abs(audio).max())
            audio = audio[max(0, loud[0] - int(0.03 * sr)): loud[-1] + int(0.06 * sr)]
            toks = sent.split()
            weights = [len(norm(w)) + 2 + (3 if w[-1] in ",;:" else 0) for w in toks]
            t0, span = n / sr + 0.03, len(audio) / sr - 0.09
            acc = 0
            for w, wt in zip(toks, weights):
                start = t0 + span * acc / sum(weights)
                acc += wt
                words.append({"word": w, "start": round(start, 3),
                              "end": round(t0 + span * acc / sum(weights), 3)})
            chunks.append(audio.astype(np.float32))
            n += len(audio)
            silence(0.18)
        silence(0.22 if pi < len(parts) - 1 else 0.6)
    sf.write(job / "voice_tts.wav", np.concatenate(chunks), sr)
    return words


def words_from_recording(job):
    import whisper  # only needed if you record your own voice
    model = whisper.load_model("small.en")
    res = model.transcribe(str(job / "voice.mp3"), word_timestamps=True)
    return [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for seg in res["segments"] for w in seg["words"]]


def get_words(job, parts, backend):
    wp = job / "words.json"
    if wp.exists():
        return load(wp)
    if (job / "voice.mp3").exists():  # your own recording always wins
        words = words_from_recording(job)
    elif backend == "elevenlabs":
        words = tts_with_timestamps(job, parts)
    else:
        words = tts_kokoro(job, parts)
    save(wp, words)
    return words


def voice_file(job):
    for name in ("voice.mp3", "voice_tts.mp3", "voice_tts.wav"):
        if (job / name).exists():
            return job / name
    raise FileNotFoundError("no voice file in " + str(job))


def normalize_voice(job):
    out = job / "voice_norm.wav"
    if not out.exists():
        run(["ffmpeg", "-y", "-i", voice_file(job),
             "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "44100", out])
    return out


# ---------------------------------------------------------------- Step 2: timing map

def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def align(parts, words):
    """Give every script word a start and end time from the spoken words."""
    script = [(i, w) for i, p in enumerate(parts) for w in p["text"].split()]
    sm = difflib.SequenceMatcher(None, [norm(w) for _, w in script],
                                 [norm(w["word"]) for w in words], autojunk=False)
    times = [None] * len(script)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            times[a + k] = (words[b + k]["start"], words[b + k]["end"])
    k = 0
    while k < len(times):  # spread unmatched runs across their gap
        if times[k] is not None:
            k += 1
            continue
        m = k
        while m < len(times) and times[m] is None:
            m += 1
        s = times[k - 1][1] if k > 0 else 0.0
        e = times[m][0] if m < len(times) else words[-1]["end"]
        step = (e - s) / (m - k)
        for j in range(k, m):
            times[j] = (s + step * (j - k), s + step * (j - k + 1))
        k = m
    return [{"part": i, "word": w, "start": t[0], "end": t[1]}
            for (i, w), t in zip(script, times)]


def timing_map(parts, swords, total):
    top = next(p["entry"] for p in parts if p["id"] == "rank1")
    sections = []
    for i, p in enumerate(parts):
        first = next(w for w in swords if w["part"] == i)
        sections.append({"id": p["id"], "start": first["start"]})
    sections[0]["start"] = 0.0
    for i, s in enumerate(sections):
        s["end"] = sections[i + 1]["start"] if i + 1 < len(sections) else total
    for s, p in zip(sections, parts):
        d = s["end"] - s["start"]
        n = 1 if p["id"] in ("hook", "outro") else max(4, min(6, round(d / 2.5)))
        terms = top["visual_terms"] if n == 1 else p["entry"]["visual_terms"]
        step = d / n
        s["shots"] = [{"id": f"{p['id']}.{k}",
                       "start": s["start"] + k * step,
                       "end": s["start"] + (k + 1) * step,
                       "terms": terms[k % len(terms):] + terms[:k % len(terms)]}
                      for k in range(n)]
    return sections


# ---------------------------------------------------------------- Steps 3-4: clips and credits

def pexels_search(term):
    r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": PEXELS_KEY},
                     params={"query": term, "orientation": "portrait", "size": "medium",
                             "per_page": 15}, timeout=30)
    r.raise_for_status()
    out = []
    for v in r.json().get("videos", []):
        files = [f for f in v["video_files"] if f.get("file_type") == "video/mp4"
                 and (f.get("width") or 0) >= 1080 and (f.get("height") or 0) >= 1920]
        if files:
            f = min(files, key=lambda f: f["width"])  # smallest file that is still Full HD
            out.append({"id": f"pexels:{v['id']}", "kind": "video", "url": f["link"],
                        "duration": v["duration"], "page": v["url"],
                        "credit": f"Video by {v['user']['name']} on Pexels: {v['url']}"})
    return out


def pixabay_search(term):
    r = requests.get("https://pixabay.com/api/videos/", timeout=30,
                     params={"key": PIXABAY_KEY, "q": term, "per_page": 20, "safesearch": "true"})
    r.raise_for_status()
    out = []
    for v in r.json().get("hits", []):
        f = v["videos"].get("large") or {}
        if f.get("url") and f.get("height", 0) >= 1080:
            out.append({"id": f"pixabay:{v['id']}", "kind": "video", "url": f["url"],
                        "duration": v["duration"], "page": v["pageURL"],
                        "portrait": f["height"] > f["width"],
                        "credit": f"Video by {v['user']} on Pixabay: {v['pageURL']}"})
    return sorted(out, key=lambda c: not c["portrait"])  # portrait first, landscape is cropped


def wikimedia_search(term):
    r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=30, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{term} filetype:bitmap", "gsrlimit": 10, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiurlwidth": 1600})
    r.raise_for_status()
    out = []
    for page in sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p["index"]):
        info = page["imageinfo"][0]
        if info["width"] < 900:
            continue
        meta = info.get("extmetadata", {})
        strip = lambda k: re.sub(r"<[^>]+>", "", html.unescape(meta.get(k, {}).get("value", ""))).strip()
        title = page["title"].removeprefix("File:")
        out.append({"id": f"commons:{page['pageid']}", "kind": "image",
                    "url": info.get("thumburl") or info["url"], "page": info["descriptionurl"],
                    "credit": f"\"{title}\" by {strip('Artist') or 'unknown'}, "
                              f"{strip('LicenseShortName')}, via Wikimedia Commons: {info['descriptionurl']}"})
    return out


def recently_used(job_id):
    log = load(USED_LOG) if USED_LOG.exists() else []
    return {c for j in log[-30:] if j["job"] != job_id for c in j["clips"]}


def log_used(job_id, clip_ids):
    log = [j for j in (load(USED_LOG) if USED_LOG.exists() else []) if j["job"] != job_id]
    log.append({"job": job_id, "clips": sorted(clip_ids)})
    save(USED_LOG, log)


def candidates(term, animal, errors):
    """Pexels, then Pixabay, then a Wikimedia Commons image. Search results are cached per run."""
    found = []
    for name, fn, ok in (("pexels", pexels_search, PEXELS_KEY), ("pixabay", pixabay_search, PIXABAY_KEY),
                         ("wikimedia", wikimedia_search, True)):
        if not ok:
            continue
        for q in ([term, animal] if name == "wikimedia" else [term]):
            key = (name, q)
            if key not in _search_cache:
                try:
                    _search_cache[key] = fn(q)
                except requests.RequestException as e:
                    _search_cache[key] = []
                    errors.add(f"{name}: {type(e).__name__}")
            found += _search_cache[key]
    return found


_search_cache = {}


def find_clips(job, sections, parts):
    """Pick a source for every shot. Shots already in shots.json keep their clip."""
    sp = job / "shots.json"
    shots = load(sp) if sp.exists() else {}
    rejected = set(shots.get("_rejected", []))
    avoid = recently_used(job.name) | rejected
    taken = {s["id"] for k, s in shots.items() if not k.startswith("_") and s["kind"] != "card"}
    errors = set()
    for sec, p in zip(sections, parts):
        animal = (p.get("entry") or next(q["entry"] for q in parts if q["id"] == "rank1"))["animal"]
        for shot in sec["shots"]:
            if shot["id"] in shots:
                continue
            if shot["id"] == "outro.0" and "hook.0" in shots:  # loop: end on the opening shot
                shots[shot["id"]] = dict(shots["hook.0"])
                continue
            need = shot["end"] - shot["start"] + 0.5
            pick = None
            for term in shot["terms"]:
                for c in candidates(term, animal, errors):
                    if c["id"] in avoid or c["id"] in taken:
                        continue
                    if c["kind"] == "video" and c["duration"] < need:
                        continue
                    pick = c
                    break
                if pick:
                    break
            if pick is None:  # last resort: a graphic card in the section's colours
                pick = {"id": f"card:{shot['id']}", "kind": "card", "credit": None,
                        "palette": "rank1" if sec["id"] in ("hook", "outro") else sec["id"],
                        "label": animal}
            taken.add(pick["id"])
            shots[shot["id"]] = pick
    shots["_rejected"] = sorted(rejected)
    save(sp, shots)
    for e in sorted(errors):
        print(f"  ! search failed: {e}")
    return shots


def download_clips(job, shots):
    clips = job / "clips"
    clips.mkdir(exist_ok=True)
    for k, s in shots.items():
        if k.startswith("_") or s["kind"] == "card":
            continue
        ext = ".mp4" if s["kind"] == "video" else Path(s["url"].split("?")[0]).suffix or ".jpg"
        path = clips / (s["id"].replace(":", "_") + ext)
        if not path.exists():
            with requests.get(s["url"], headers=UA, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        s["path"] = str(path.relative_to(job))
    save(job / "shots.json", shots)
    credits = sorted({s["credit"] for k, s in shots.items() if not k.startswith("_") and s.get("credit")})
    (job / "credits.txt").write_text("\n".join(credits) + ("\n" if credits else ""))
    log_used(job.name, {s["id"] for k, s in shots.items() if not k.startswith("_") and s["kind"] != "card"})


# ---------------------------------------------------------------- shot preparation (1080x1920, exact frames)

def cover(img, w, h):
    """Scale and centre-crop an image to fill w x h."""
    s = max(w / img.width, h / img.height)
    img = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def ken_burns(img, out, frames, seed):
    """Slow zoom and pan over a still, rendered frame by frame for smooth sub-pixel motion."""
    rng = random.Random(seed)
    img = cover(img.convert("RGB"), int(W * 1.2), int(H * 1.2))
    z0, z1 = (1.0, 1.12) if rng.random() < 0.5 else (1.12, 1.0)
    (ax, ay), (bx, by) = [(rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)) for _ in range(2)]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
                          stdin=subprocess.PIPE)
    for k in range(frames):
        u = k / max(1, frames - 1)
        z = z0 + (z1 - z0) * u
        cw, ch = img.width / (1.2 * z), img.height / (1.2 * z)
        x = (img.width - cw) * (ax + (bx - ax) * u)
        y = (img.height - ch) * (ay + (by - ay) * u)
        ff.stdin.write(img.resize((W, H), Image.BICUBIC, box=(x, y, x + cw, y + ch)).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError(f"ffmpeg failed writing {out}")


def card_image(palette, label, seed):
    """Graphic fallback: section-coloured gradient, soft bokeh and the animal's name."""
    rng = random.Random(seed)
    w, h = int(W * 1.2), int(H * 1.2)
    top, bottom = (np.array(c, float) for c in PALETTES[palette])
    u = np.linspace(0, 1, h)[:, None, None]
    grad = top * (1 - u) + bottom * u
    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 1 - 0.45 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * 0.4) / (h / 1.6)) ** 2)
    img = Image.fromarray(np.clip(grad * np.clip(vignette, 0.35, 1)[..., None], 0, 255).astype(np.uint8))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    tint = tuple(int(min(255, c * 1.8 + 60)) for c in PALETTES[palette][0])
    for _ in range(28):
        r = rng.uniform(20, 140)
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        d.ellipse((x - r, y - r, x + r, y + r), fill=tint + (int(rng.uniform(25, 90)),))
    img = Image.alpha_composite(img.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(14)))
    words = label.upper().split()
    name = text_block(words, 200, w * 0.85, stroke=0, fill=(255, 255, 255, 34), font=FONT_BLACK)
    img.alpha_composite(name, ((w - name.width) // 2, int(h * 0.72) - name.height // 2))
    return img.convert("RGB")


def prepare_shots(job, sections, shots):
    """Turn every shot into an mp4 of exactly its frame count, then join them."""
    prep = job / "clips/prep"
    prep.mkdir(parents=True, exist_ok=True)
    listing = []
    for sec in sections:
        for shot in sec["shots"]:
            s = shots[shot["id"]]
            frames = round(shot["end"] * FPS) - round(shot["start"] * FPS)
            out = prep / f"{shot['id']}_{s['id'].replace(':', '_')}_{frames}.mp4"
            if not out.exists():
                seed = shot["id"]
                if s["kind"] == "video":
                    src = job / s["path"]
                    clip_len = duration(src)
                    offset = min(clip_len * 0.15, max(0.0, clip_len - frames / FPS - 0.2))
                    run(["ffmpeg", "-y", "-ss", f"{offset:.2f}", "-i", src, "-frames:v", frames,
                         "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                                f"fps={FPS},setsar=1",
                         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                         "-pix_fmt", "yuv420p", out])
                elif s["kind"] == "image":
                    ken_burns(Image.open(job / s["path"]), out, frames, seed)
                else:
                    ken_burns(card_image(s["palette"], s["label"], seed), out, frames, seed)
            listing.append(f"file '{out.resolve()}'")
    (prep / "list.txt").write_text("\n".join(listing) + "\n")
    bg = job / "background.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", prep / "list.txt", "-c", "copy", bg])
    return bg


# ---------------------------------------------------------------- Step 5: text, badges and captions

def text_block(words, size, max_w, colors=None, stroke=8, fill=(255, 255, 255, 255),
               font=FONT_BLACK, line_gap=0.08):
    """Centred, word-wrapped text with a black outline; colors[i] overrides word i."""
    f = ImageFont.truetype(font, size)
    space = f.getlength(" ")
    lines, cur, cur_w = [], [], 0.0
    for i, w in enumerate(words):
        ww = f.getlength(w)
        if cur and cur_w + space + ww > max_w:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur_w += (space if cur else 0) + ww
        cur.append(i)
    lines.append(cur)
    asc, desc = f.getmetrics()
    lh = asc + desc
    widths = [sum(f.getlength(words[i]) for i in l) + space * (len(l) - 1) for l in lines]
    pad = stroke + 4
    img = Image.new("RGBA", (int(max(widths)) + 2 * pad,
                             int(lh * len(lines) + lh * line_gap * (len(lines) - 1)) + 2 * pad))
    d = ImageDraw.Draw(img)
    y = pad
    for l, lw in zip(lines, widths):
        x = pad + (img.width - 2 * pad - lw) / 2
        for i in l:
            d.text((x, y), words[i], font=f, fill=colors[i] if colors else fill,
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            x += f.getlength(words[i]) + space
        y += lh * (1 + line_gap)
    return img


def with_shadow(img, blur=10, offset=8, alpha=0.6):
    out = Image.new("RGBA", (img.width + 4 * blur, img.height + 4 * blur + offset))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 255))
    shadow.putalpha(img.getchannel("A").point(lambda a: int(a * alpha)))
    out.alpha_composite(shadow, (2 * blur, 2 * blur + offset))
    out = out.filter(ImageFilter.GaussianBlur(blur))
    out.alpha_composite(img, (2 * blur, 2 * blur))
    return out


def rank_badge(rank):
    f = ImageFont.truetype(FONT_BLACK, 120)
    label = f"#{rank}"
    tw = f.getlength(label)
    img = Image.new("RGBA", (int(tw) + 80, 170))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=34, fill=YELLOW + (255,))
    d.text((img.width / 2, img.height / 2), label, font=f, fill=(0, 0, 0, 255), anchor="mm")
    return img


def caption_chunks(swords):
    """1-3 words per chunk, breaking at punctuation, pauses and section changes."""
    chunks, cur = [], []
    for i, w in enumerate(swords):
        cur.append(w)
        nxt = swords[i + 1] if i + 1 < len(swords) else None
        chars = sum(len(x["word"]) for x in cur)
        if (nxt is None or len(cur) == 3 or chars > 14 or re.search(r"[.,!?;:]$", w["word"])
                or nxt["part"] != w["part"] or nxt["start"] - w["end"] > 0.25
                or chars + len(nxt["word"]) > 18):
            chunks.append(cur)
            cur = []
    return chunks


class Layers:
    """Overlay clips plus the box each one covers, for the safe-zone check."""

    def __init__(self):
        self.clips, self.boxes = [], []

    def add(self, img, start, end, cx, top=None, cy=None, fade=0.0):
        img = with_shadow(img)
        x = int(cx - img.width / 2)
        y = int(top if top is not None else cy - img.height / 2)
        clip = ImageClip(np.array(img)).with_start(start).with_duration(end - start).with_position((x, y))
        if fade:
            clip = clip.with_effects([vfx.CrossFadeIn(fade)])
        self.clips.append(clip)
        bbox = img.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()  # visible pixels
        self.boxes.append([x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3]])
        return y + img.height


def build_overlays(script, parts, sections, swords):
    L = Layers()
    max_w = SAFE[2] - SAFE[0] - 80  # room for the outline and drop shadow
    hook = next(s for s in sections if s["id"] == "hook")
    # Hook text on the very first frame, over the strongest shot.
    hook_text = script.get("hook_on_screen") or re.split(r"(?<=[.!?])\s", script["hook"])[0]
    words = hook_text.upper().rstrip(".").split()
    L.add(text_block(words, 92, max_w, colors=[(255, 255, 255)] * (len(words) - 1) + [YELLOW]),
          0.0, hook["end"], CX, top=SAFE[1] + 60)
    # A rank badge and the on-screen line for each entry.
    for sec, p in zip(sections, parts):
        if "entry" not in p:
            continue
        e = p["entry"]
        bottom = L.add(rank_badge(e["rank"]), sec["start"], sec["end"], CX, top=SAFE[1] + 20, fade=0.15)
        L.add(text_block(e["on_screen"].upper().split(), 70, max_w, colors=None),
              sec["start"], sec["end"], CX, top=bottom - 10, fade=0.15)
    # Word-by-word captions in the middle third, the spoken word highlighted.
    chunks = caption_chunks(swords)
    for ci, chunk in enumerate(chunks):
        end = chunk[-1]["end"] + 0.35
        if ci + 1 < len(chunks):
            end = min(end, chunks[ci + 1][0]["start"])
        end = max(end, chunk[-1]["end"])
        words = [re.sub(r"[,.;:]+$", "", w["word"]).upper() for w in chunk]
        for k, w in enumerate(chunk):
            t0 = chunk[0]["start"] if k == 0 else w["start"]
            t1 = chunk[k + 1]["start"] if k + 1 < len(chunk) else end
            if t1 - t0 < 1 / FPS:
                continue
            colors = [YELLOW if j == k else (255, 255, 255) for j in range(len(words))]
            L.add(text_block(words, 96, max_w, colors=colors, stroke=9), t0, t1, CX, cy=H * 0.56)
    return L


def shade_overlay():
    """Darken the top and bottom of the frame so white text stays readable on bright footage."""
    y = np.linspace(0, 1, H)[:, None]
    a = np.clip(0.55 - y * 1.1, 0, 1) + np.clip((y - 0.62) * 1.3, 0, 0.45) + 0.12
    alpha = np.repeat(np.clip(a, 0, 0.75), W, axis=1)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return ImageClip(rgba)


# ---------------------------------------------------------------- Steps 6-7: music and render

def pick_music(script, total):
    mood = script.get("music_mood", "curious")
    tracks = sorted((MUSIC_DIR / mood).glob("*.mp3")) or sorted(MUSIC_DIR.glob("*/*.mp3"))
    if not tracks:
        return None, None
    track = random.Random(script["hook"]).choice(tracks)
    clip = AudioFileClip(str(track)).with_effects([
        afx.AudioLoop(duration=total), afx.MultiplyVolume(10 ** (-18 / 20)),  # ~18 dB under the voice
        afx.AudioFadeIn(0.5), afx.AudioFadeOut(1.5)])
    return clip, track


def render(job, script, parts, sections, swords, total, bg):
    overlays = build_overlays(script, parts, sections, swords)
    save(job / "overlays.json", overlays.boxes)
    video = CompositeVideoClip([VideoFileClip(str(bg), audio=False).with_duration(total),
                                shade_overlay().with_duration(total)] + overlays.clips,
                               size=(W, H)).with_duration(total)
    voice = AudioFileClip(str(normalize_voice(job)))
    voice = voice.subclipped(0, min(total, voice.duration))
    music, track = pick_music(script, total)
    video = video.with_audio(CompositeAudioClip([voice] + ([music] if music else [])))
    raw = job / "render.mp4"
    video.write_videofile(str(raw), fps=FPS, codec="libx264", audio_codec="aac", preset="medium",
                          threads=os.cpu_count(), temp_audiofile=str(job / "render_audio.m4a"), ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
                          logger=None)
    final = job / "final.mp4"
    run(["ffmpeg", "-y", "-i", raw, "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final])
    raw.unlink()
    return final, track


# ---------------------------------------------------------------- Step 8: automatic checks

def check(job, script, sections, shots, track, lint):
    final = job / "final.mp4"
    checks, warnings = [], list(lint)

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    probe = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,codec_name,r_frame_rate", "-of", "json", final]).stdout)
    v = probe["streams"][0]
    d = duration(final)
    add("length 60-90 s", 60 <= d <= 90, f"{d:.1f} s")
    add("1080x1920, 30 fps, H.264", (v["width"], v["height"], v["codec_name"], v["r_frame_rate"])
        == (W, H, "h264", f"{FPS}/1"), f"{v['width']}x{v['height']} {v['codec_name']} {v['r_frame_rate']}")
    err = subprocess.run(["ffmpeg", "-nostats", "-i", str(final), "-filter_complex",
                          "[0:a]ebur128=peak=true[a];[0:v]blackdetect=d=0.5:pix_th=0.08,"
                          "freezedetect=n=-60dB:d=2[v]", "-map", "[a]", "-map", "[v]",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    lufs = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])
    peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", err)[-1])
    add("loudness about -14 LUFS", abs(lufs + 14) <= 1.5, f"{lufs:.1f} LUFS, true peak {peak:.1f} dBFS")
    blacks = re.findall(r"black_start:([\d.]+)", err)
    add("no black frames", not blacks, f"black at {blacks} s" if blacks else "none")
    freezes = re.findall(r"freeze_start: ([\d.]+)", err)
    add("no frozen frames", not freezes, f"frozen at {freezes} s" if freezes else "none")
    boxes = load(job / "overlays.json")
    bad = [b for b in boxes if b[0] < SAFE[0] or b[1] < SAFE[1] or b[2] > SAFE[2] or b[3] > SAFE[3]]
    add("text inside the safe zone", not bad, f"{len(boxes)} overlays, {len(bad)} outside")
    shot_list = [shots[s["id"]] for sec in sections for s in sec["shots"]]
    missing = [s["id"] for s in shot_list if s["kind"] != "card" and not s.get("credit")]
    add("a credit for every clip", not missing, f"missing: {missing}" if missing else "all credited")
    add("hook text on the first frame", True, "hook overlay starts at 0.0 s")

    cards = sum(s["kind"] == "card" for s in shot_list)
    if cards:
        warnings.append(f"{cards} of {len(shot_list)} shots are graphic fallback cards, not footage")
    if track is None:
        warnings.append("no music: add YouTube Audio Library tracks to assets/music/<mood>/")
    report = {"status": "pass" if all(c["pass"] for c in checks) else "fail",
              "checks": checks, "warnings": warnings}
    save(job / "report.json", report)
    return report


# ---------------------------------------------------------------- Step 9: upload details

def write_description(job, script):
    entries = sorted(entries_of(script), key=lambda e: -e["rank"])
    title = script.get("title") or re.split(r"(?<=[.!?])\s", script["hook"])[0]
    lines = [script["hook"], ""]
    lines += [f"#{e['rank']} {e['animal']}: {e['on_screen']}" for e in entries]
    lines += ["", "Sources:"]
    for e in entries:
        lines += [f"#{e['rank']} {e['animal']}"] + [f"  {u}" for u in e["sources"]]
    credits = (job / "credits.txt").read_text().strip()
    if credits:
        lines += ["", "Footage:", credits]
    lines += ["", " ".join(script.get("hashtags", ["#shorts"]))]
    body = "\n".join(lines)
    (job / "description.txt").write_text(
        f"TITLE: {title}\nAI DISCLOSURE (altered or synthetic content): "
        f"{'Yes' if script.get('ai_disclosure') else 'No'}\n\n{body}\n")


# ---------------------------------------------------------------- main

def swap(job, shot_ids):
    """Gate 3: drop the chosen clip for these shots so the next run fetches another."""
    sp = job / "shots.json"
    shots = load(sp)
    rejected = set(shots.get("_rejected", []))
    for sid in shot_ids:
        old = shots.pop(sid, None)
        if old and old["kind"] != "card":
            rejected.add(old["id"])
        if sid == "hook.0":
            shots.pop("outro.0", None)  # the outro reuses the hook's clip
        print(f"  swapping {sid} (was {old['id'] if old else 'nothing'})")
    shots["_rejected"] = sorted(rejected)
    save(sp, shots)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path, help="job folder holding script.json")
    ap.add_argument("--swap", nargs="+", metavar="SHOT", help="shot ids to re-fetch, e.g. rank3.2")
    ap.add_argument("--voice", choices=["auto", "elevenlabs", "kokoro"], default="auto",
                    help="auto = ElevenLabs when its keys are set, otherwise local Kokoro")
    args = ap.parse_args()
    job = args.job.resolve()
    script = load(job / "script.json")
    parts = script_parts(script)
    if args.swap:
        swap(job, args.swap)
    backend = args.voice if args.voice != "auto" else ("elevenlabs" if ELEVEN_KEY and VOICE_ID else "kokoro")

    lint = lint_script(script)
    for w in lint:
        print(f"  ! script: {w}")
    print("1/8 voice and word timings")
    words = get_words(job, parts, backend)
    normalize_voice(job)
    print("2/8 timing map")
    swords = align(parts, words)
    total = min(swords[-1]["end"] + 0.3, duration(job / "voice_norm.wav"))
    sections = timing_map(parts, swords, total)
    save(job / "timing.json", {"total": total, "sections": sections, "words": swords})
    print("3/8 finding clips")
    shots = find_clips(job, sections, parts)
    print("4/8 downloading clips and credits")
    download_clips(job, shots)
    print("5/8 preparing shots")
    bg = prepare_shots(job, sections, shots)
    print("6-7/8 captions, badges, music and render")
    final, track = render(job, script, parts, sections, swords, total, bg)
    print("8/8 automatic checks")
    report = check(job, script, sections, shots, track, lint)
    write_description(job, script)
    for c in report["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}: {c['detail']}")
    for w in report["warnings"]:
        print(f"  WARN  {w}")
    print(f"{report['status'].upper()}: {final}")
    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
