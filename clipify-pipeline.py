from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import shutil
import subprocess
import sys
import json
import os
import wave
import struct
import math
import base64

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_FILE      = "audio.mp3"
CLIPS_DIR       = "clips"
MODEL_SIZE      = "base"     # 3-4x faster than "small"; accurate enough for highlight detection

MIN_SEG_SCORE   = 2          # min score for a segment to be "interesting"
CLUSTER_GAP     = 30         # seconds: high-scoring segs within this → one clip
TARGET_DURATION = 60         # ideal clip length (seconds) — overridden by preset below
MIN_DURATION    = 20         # discard clips shorter than this
MAX_DURATION    = 90         # cap clips at this
CLIP_PADDING    = 2.0        # seconds to extend before/after each clip zone
MIN_ZONE_SCORE  = 5          # minimum cumulative score for a zone to qualify

FAST_MODE         = True   # skip caption burn-in, use stream copy (much faster)
MAX_SCAN_SEGMENTS = 5000   # hard cap on segments scanned to prevent O(n) blowup

# ── Clip-length presets ───────────────────────────────────────────────────────
DURATION_PRESETS = {
    #              target  min  max  cluster_gap
    "short":  dict(target=25,  min_dur=15, max_dur=35, cluster_gap=20),
    "medium": dict(target=50,  min_dur=25, max_dur=70, cluster_gap=30),
    "long":   dict(target=78,  min_dur=58, max_dur=90, cluster_gap=40),
}

analyzer = SentimentIntensityAnalyzer()

# ── Global Whisper model cache (loaded once per process) ──────────────────────
_WHISPER_TINY: "WhisperModel | None" = None
_WHISPER_BASE: "WhisperModel | None" = None

def _get_whisper(size: str) -> "WhisperModel":
    global _WHISPER_TINY, _WHISPER_BASE
    cpu_threads = min(os.cpu_count() or 2, 8)
    if size == "tiny":
        if _WHISPER_TINY is None:
            print(f"Loading Whisper model (tiny)...", flush=True)
            _WHISPER_TINY = WhisperModel("tiny", compute_type="int8", cpu_threads=cpu_threads)
        return _WHISPER_TINY
    else:
        if _WHISPER_BASE is None:
            print(f"Loading Whisper model ({MODEL_SIZE})...", flush=True)
            _WHISPER_BASE = WhisperModel(MODEL_SIZE, compute_type="int8", cpu_threads=cpu_threads)
        return _WHISPER_BASE

# ── Swear-word censoring (stored as an opaque blob — no plaintext in source) ──
_CDB = json.loads(base64.b64decode(
    "eyJmdWNrIjoiZioqayIsImZ1Y2tpbmciOiJmKipraW5nIiwiZnVja2VkIjoiZioqa2VkIiwi"
    "ZnVja2VyIjoiZioqa2VyIiwiZnVja3MiOiJmKiprcyIsInNoaXQiOiJzaCp0Iiwic2hpdHMi"
    "OiJzaCp0cyIsInNoaXR0aW5nIjoic2gqdHRpbmciLCJzaGl0dHkiOiJzaCp0dHkiLCJidWxs"
    "c2hpdCI6ImJ1bGxzKip0IiwiYml0Y2giOiJiKipjaCIsImJpdGNoZXMiOiJiKipjaGVzIiwi"
    "YXNzIjoiYSoqIiwiYXNzZXMiOiJhKiplcyIsImFzc2hvbGUiOiJhKipob2xlIiwiYXNzaG9s"
    "ZXMiOiJhKipob2xlcyIsImRpY2siOiJkKiprIiwiZGlja3MiOiJkKiprcyIsImNvY2siOiJj"
    "KiprIiwiY29ja3MiOiJjKiprcyIsImN1bnQiOiJjKip0IiwiY3VudHMiOiJjKip0cyIsInNl"
    "eCI6InMqeCIsInNleHkiOiJzKnh5Iiwic2V4dWFsIjoicyp4dWFsIiwicGlzcyI6InAqc3Mi"
    "LCJwaXNzZWQiOiJwKnNzZWQiLCJzbHV0Ijoic2wqdCIsInNsdXRzIjoic2wqdHMiLCJ3aG9y"
    "ZSI6IndoKnJlIiwid2hvcmVzIjoid2gqcmVzIiwiYmFzdGFyZCI6ImIqc3RhcmQiLCJiYXN0"
    "YXJkcyI6ImIqc3RhcmRzIiwiZmFnIjoiZipnIiwiZmFnZ290IjoiZioqKip0IiwibmlnZ2Vy"
    "IjoibioqKioqIiwibmlnZ2EiOiJuKioqYSIsImRhbW4iOiJkKm1uIiwiZGFtbmVkIjoiZCpt"
    "bmVkIiwiaGVsbCI6ImgqbGwiLCJjcmFwIjoiY3IqcCIsInJldGFyZCI6InIqKioqZCIsInJl"
    "dGFyZGVkIjoicioqKipkZWQifQ=="
))
_CENSOR_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _CDB) + r')\b',
    re.IGNORECASE,
)

def censor(text: str) -> str:
    return _CENSOR_RE.sub(lambda m: _CDB.get(m.group(0).lower(), m.group(0)), text)


# ── Audio extraction ──────────────────────────────────────────────────────────
def extract_audio_for_transcription(video_path: str) -> str:
    """
    Pull the audio track out as a 16 kHz mono WAV — Whisper's native format.
    Avoids Whisper having to demux the full video stream, which alone can cut
    transcription time by 20-40% on large files.
    Falls back to the original path if ffmpeg fails.
    """
    wav_path = video_path + "_whisper.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",           # drop video stream
        "-ar", "16000",  # 16 kHz — Whisper's native sample rate
        "-ac", "1",      # mono
        "-f", "wav",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            return wav_path
    except Exception:
        pass
    print("  audio extraction failed, falling back to original file", flush=True)
    return video_path


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe(path: str, word_timestamps: bool = False):
    # tiny for scoring-only pass (no word timestamps), base for annotation
    model = _get_whisper("base" if word_timestamps else "tiny")
    print("Transcribing...", flush=True)
    segments, _ = model.transcribe(
        path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=word_timestamps,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return list(segments)


# ── Hook generation ───────────────────────────────────────────────────────────
def _extract_subject(text: str) -> str:
    """Detect a short subject phrase from transcript text for hook templates."""
    lower = " " + text.lower() + " "
    for phrase in ("this guy", "this dude", "this man", "this woman", "this person"):
        if phrase in lower:
            return phrase.upper()
    for token, label in ((" he ", "HE"), (" she ", "SHE"), (" they ", "THEY"), (" i ", "I")):
        if token in lower:
            return label
    return "THIS GUY"


def generate_clip_title(text: str, score: float = 0.0) -> str:
    """Return a punchy, curiosity-driven uppercase hook title (5–10 words)."""
    lower = text.lower()
    subject = _extract_subject(text)
    sentiment = analyzer.polarity_scores(text)["compound"]

    # Keyword-driven template selection (deterministic, ordered by specificity)
    if any(p in lower for p in ("can't believe", "cannot believe", "i never")):
        title = f"I CAN'T BELIEVE {subject} DID THIS"
    elif any(w in lower for w in ("crazy", "insane", "unbelievable", "mind-blowing")):
        title = f"{subject} SAID THE CRAZIEST THING"
    elif any(w in lower for w in ("mistake", "disaster", "ruined", "backfired")):
        title = "THIS WAS A HUGE MISTAKE"
    elif any(w in lower for w in ("secret", "exposed", "nobody knows", "they don't want")):
        title = "NOBODY TALKS ABOUT THIS"
    elif any(w in lower for w in ("changed", "transformation", "changed my life", "game changer")):
        title = "THIS CHANGED EVERYTHING"
    elif any(h in lower for h in ("no way", "you won't believe", "wait until")):
        title = f"NO WAY {subject} JUST DID THIS"
    elif any(w in lower for w in ("shocking", "jaw-dropping", "impossible", "never seen")):
        title = f"WAIT UNTIL YOU SEE WHAT {subject} DID"
    elif sentiment < -0.35 or any(w in lower for w in ("arrested", "betrayed", "fraud", "scam")):
        title = "THE TRUTH FINALLY CAME OUT"
    elif score > 12 or any(w in lower for w in ("million", "billion", "greatest ever", "biggest")):
        title = "YOU WON'T BELIEVE THIS"
    else:
        # Fallback: pick punchiest sentence, trim to 7 words
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.split()) >= 3]
        if not sentences:
            return "THIS IS ABSOLUTELY INSANE"
        best = max(sentences, key=lambda s: (
            sum(3 for w in HIGH_IMPACT   if w in s.lower()) +
            sum(2 for w in MEDIUM_IMPACT if w in s.lower())
        ))
        words = best.split()[:7]
        title = " ".join(words).upper()

    return title


def _thumbnail_label(title: str, score: float = 0.0) -> str:
    """Shorten title to max 5 words for thumbnail overlay; append intensity emoji."""
    lower = title.lower()
    if any(k in lower for k in ("can't believe", "insane", "crazy", "unbelievable", "no way")):
        emoji = " 😳"
    elif any(k in lower for k in ("mistake", "disaster", "exposed", "truth", "fraud")):
        emoji = " 😱"
    elif score > 10 or any(k in lower for k in ("wild", "huge", "biggest", "greatest", "changed")):
        emoji = " 🔥"
    else:
        emoji = ""
    words = title.replace("...", "").split()[:5]
    return " ".join(words) + emoji


def generate_clip_summary(text: str, score: float = 0.0) -> str:
    """Return a punchy one-liner hook — not an extractive transcript summary."""
    lower = text.lower()
    sentiment = analyzer.polarity_scores(text)["compound"]

    if any(p in lower for p in ("can't believe", "unbelievable", "i never")):
        return "The moment that left everyone speechless."
    elif any(w in lower for w in ("crazy", "insane", "mind-blowing")):
        return "Things got crazy fast — you have to see this."
    elif any(w in lower for w in ("secret", "exposed", "nobody knows")):
        return "The secret they never wanted you to find out."
    elif any(w in lower for w in ("mistake", "disaster", "backfired")):
        return "A mistake that changed everything."
    elif any(w in lower for w in ("changed", "transformation", "game changer")):
        return "The moment everything changed."
    elif sentiment > 0.4:
        return "This had everyone talking."
    elif sentiment < -0.35:
        return "Nobody saw this coming."
    elif score > 10:
        return "The wildest moment of the entire video."
    else:
        return "One of the most talked-about moments in the clip."


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_ts(s: float) -> str:
    t = int(s)
    return f"{t // 60:02d}:{t % 60:02d}"

def safe_ts(s: float) -> str:
    return fmt_ts(s).replace(":", "-")

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── Audio energy ──────────────────────────────────────────────────────────────
def get_segment_energy(wav_path: str, start: float, end: float) -> float:
    """
    Compute normalised RMS energy (0–1) for a time slice of a 16 kHz mono WAV.
    Loud segments = raised/excited voice; this adds a hard-to-fake signal on top
    of the keyword/sentiment scoring.
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            rate   = wf.getframerate()
            n      = max(0, int((end - start) * rate))
            offset = max(0, int(start * rate))
            wf.setpos(offset)
            raw = wf.readframes(n)
        if len(raw) < 2:
            return 0.0
        n_samples = len(raw) // 2
        samples   = struct.unpack(f"<{n_samples}h", raw[: n_samples * 2])
        rms       = math.sqrt(sum(s * s for s in samples) / n_samples)
        return min(rms / 32768.0, 1.0)
    except Exception:
        return 0.0


# ── Scoring (applied per-segment) ─────────────────────────────────────────────
HIGH_IMPACT = [
    "crazy", "insane", "unbelievable", "incredible", "mind-blowing",
    "jaw-dropping", "shocking", "impossible", "never seen", "first time",
    "destroyed", "exposed", "secret", "nobody knows", "revealed", "disaster",
    "died", "survived", "broke", "scam", "fraud", "betrayed", "arrested",
    "million", "billion", "biggest", "worst ever", "greatest ever",
]
MEDIUM_IMPACT = [
    "amazing", "awesome", "wow", "huge", "massive", "problem", "mistake",
    "truth", "actually", "literally", "wild", "ridiculous", "important",
    "surprising", "unexpected", "won", "lost", "love", "hate", "angry",
    "scared", "excited", "money", "rich", "free", "no way", "wtf",
    "wrong", "right", "best", "worst", "fail", "change", "serious",
    "transformation", "result", "before", "after", "look at",
]
HOOKS = [
    "wait", "listen", "watch this", "look at this", "check this out",
    "here's why", "the reason", "what if", "turns out", "plot twist",
    "game changer", "i can't believe", "you won't believe", "let me show",
    "this is why", "spoiler", "breaking", "oh my", "oh no", "oh wow",
    "nobody talks about", "they don't want you to know", "i never",
    "for the first time", "it's that time",
    # Emotional reactions
    "oh my god", "oh my gosh", "holy", "no way", "wait what",
    "are you kidding", "are you serious", "i swear", "i promise",
    "trust me", "believe it or not", "you have to see", "you need to hear",
    "this changed", "changed my life", "changed everything",
    "never again", "just realized", "just found out", "found out",
    "breaking news", "can you believe", "i was shaking", "i was crying",
    "it hit me", "blew my mind", "lost my mind",
]
STRUCTURE = [
    "first", "second", "third", "finally", "number one", "step",
    "tip", "rule", "most important", "the real reason", "top",
]

# ── Hook / header config ───────────────────────────────────────────────────────
HEADER_ENABLED  = True   # burn punchy title at clip start
HEADER_DURATION = 3      # seconds the header stays visible
HEADER_POSITION = "top"  # "top" or "center"

HOOK_TEMPLATES = [
    "I CAN'T BELIEVE {SUBJECT} DID THIS",
    "{SUBJECT} SAID THE CRAZIEST THING",
    "NO WAY {SUBJECT} JUST DID THIS",
    "THIS WAS A HUGE MISTAKE",
    "NOBODY TALKS ABOUT THIS",
    "THIS CHANGED EVERYTHING",
    "WAIT UNTIL YOU SEE WHAT {SUBJECT} DID",
    "YOU WON'T BELIEVE THIS",
    "THE TRUTH FINALLY CAME OUT",
    "THIS IS ABSOLUTELY INSANE",
]


def score_segment(text: str, duration: float):
    if not text:
        return 0.0, []
    lower = text.lower()
    score = 0.0
    reasons = []

    # VADER sentiment — both extremes are engaging
    v = analyzer.polarity_scores(text)
    compound = abs(v["compound"])
    if compound >= 0.7:
        score += 6;  reasons.append(f"strong emotion ({compound:.2f})")
    elif compound >= 0.5:
        score += 4;  reasons.append(f"clear emotion ({compound:.2f})")
    elif compound >= 0.3:
        score += 2;  reasons.append(f"mild emotion ({compound:.2f})")

    if v["neg"] > 0.25:
        score += 2;  reasons.append("tension")
    if v["pos"] > 0.35:
        score += 2;  reasons.append("enthusiasm")

    for w in HIGH_IMPACT:
        if w in lower:
            score += 3;  reasons.append(f'"{w}"')
    for w in MEDIUM_IMPACT:
        if w in lower:
            score += 2;  reasons.append(f'"{w}"')
    for h in HOOKS:
        if h in lower:
            score += 3;  reasons.append(f'hook:"{h}"')
    for s in STRUCTURE:
        if s in lower:
            score += 1;  reasons.append(f'structure:"{s}"')

    excl = text.count("!")
    ques = text.count("?")
    if excl:
        score += min(excl * 2, 6);  reasons.append(f"{excl} exclamation(s)")
    if ques:
        score += min(ques * 1.5, 4); reasons.append(f"{ques} question(s)")

    words = len(text.split())
    if duration > 0:
        pace = words / duration
        if pace > 3.0:
            score += 3;  reasons.append(f"fast pace ({pace:.1f}w/s)")
        elif pace > 2.2:
            score += 1.5; reasons.append(f"active pace ({pace:.1f}w/s)")

    # Intensifiers and emphasis words
    if re.search(r"\b(really|very|so|extremely|absolutely|totally|completely|literally|honestly)\b", lower):
        score += 1;  reasons.append("emphasis")

    # All-caps words signal shouting/strong emotion
    caps = len(re.findall(r'\b[A-Z]{2,}\b', text))
    if caps:
        score += min(caps, 3);  reasons.append(f"{caps} caps word(s)")

    # Repeated consecutive words ("no no no", "stop stop") = strong emotion
    word_list = lower.split()
    for j in range(len(word_list) - 1):
        if word_list[j] == word_list[j + 1] and len(word_list[j]) > 2:
            score += 3;  reasons.append("repeated emphasis"); break

    # Swear words are almost always emotionally charged moments
    for sw in _CDB:
        if re.search(r'\b' + re.escape(sw) + r'\b', lower):
            score += 4;  reasons.append(f'profanity:"{sw}"'); break

    # Laughter, crying, or gasping captured in transcript
    if re.search(r'\b(laugh|laughing|laughter|crying|sobbing|gasp|screaming|yelling|shouting)\b', lower):
        score += 3;  reasons.append("audible emotion")

    # Direct audience address creates engagement
    if re.search(r'\b(you guys|everyone|everybody|let me tell you|i need to tell)\b', lower):
        score += 2;  reasons.append("direct address")

    if words < 5:
        score -= 3

    return max(round(score, 1), 0.0), reasons


# ── Cluster high-scoring segments into clip zones ─────────────────────────────
def find_clip_zones(segments, audio_path: str = None, min_zone_score: float = None,
                    max_clips: int = 0):
    """
    1. Score every segment individually (text + audio energy).
    2. Collect segments that score >= MIN_SEG_SCORE.
    3. Cluster those segments: if two are within CLUSTER_GAP seconds, same clip.
    4. Hard-clamp each cluster to [MIN_DURATION, MAX_DURATION] (no target expansion).
    5. Discard zones below min_zone_score (defaults to MIN_ZONE_SCORE) or shorter than MIN_DURATION.
    """
    if min_zone_score is None:
        min_zone_score = MIN_ZONE_SCORE

    # Cap segments to avoid O(n) blowup on very long transcripts
    n_total = len(segments)
    if n_total > MAX_SCAN_SEGMENTS:
        print(f"  ⚡ Capping scan: {n_total} → {MAX_SCAN_SEGMENTS} segments", flush=True)
        segments = segments[:MAX_SCAN_SEGMENTS]
    n_scanned = len(segments)

    use_energy = (audio_path is not None
                  and audio_path.endswith(".wav")
                  and os.path.exists(audio_path))

    scored_segs = []
    for seg in segments:
        text = clean(seg.text)
        score, reasons = score_segment(text, seg.end - seg.start)

        # Audio energy bonus — loud/raised voice is a strong engagement signal
        if use_energy:
            energy = get_segment_energy(audio_path, seg.start, seg.end)
            if energy > 0.35:
                score += 6;  reasons.append(f"very loud ({energy:.2f})")
            elif energy > 0.22:
                score += 3;  reasons.append(f"loud ({energy:.2f})")
            elif energy > 0.14:
                score += 1;  reasons.append(f"elevated volume ({energy:.2f})")

        scored_segs.append({
            "start":   seg.start,
            "end":     seg.end,
            "text":    text,
            "score":   score,
            "reasons": reasons,
        })

    interesting = [s for s in scored_segs if s["score"] >= MIN_SEG_SCORE]

    if not interesting:
        return []

    interesting.sort(key=lambda x: x["start"])

    print(f"  📊 {n_scanned} segments scanned, {len(interesting)} high-scoring", flush=True)

    # Build clusters; early-exit once we have 2× requested clips (enough to dedup down)
    clusters    = [[interesting[0]]]
    early_exit  = False
    early_limit = max_clips * 2 if max_clips > 0 else 0
    for seg in interesting[1:]:
        if seg["start"] - clusters[-1][-1]["end"] <= CLUSTER_GAP:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])
        if early_limit and len(clusters) >= early_limit:
            early_exit = True
            break

    if early_exit:
        print(f"  ⚡ Early exit: {len(clusters)} clusters built (requested {max_clips})", flush=True)

    zones = []
    for cluster in clusters:
        zone_start = cluster[0]["start"]
        zone_end   = cluster[-1]["end"]
        zone_dur   = zone_end - zone_start
        zone_score = sum(s["score"] for s in cluster)
        zone_text  = censor(clean(" ".join(s["text"] for s in cluster)))
        reasons    = []
        for s in cluster:
            reasons.extend(s["reasons"])
        reasons = list(dict.fromkeys(reasons))

        if zone_score < min_zone_score:
            continue

        # Strict hard clamp — no expansion toward TARGET_DURATION
        start = max(0, zone_start - CLIP_PADDING)
        end   = zone_end + CLIP_PADDING
        dur   = end - start

        if dur < MIN_DURATION:
            end = start + MIN_DURATION
            dur = MIN_DURATION

        if dur > MAX_DURATION:
            peak  = max(cluster, key=lambda x: x["score"])
            mid   = (peak["start"] + peak["end"]) / 2
            start = max(0, mid - MAX_DURATION / 2)
            end   = start + MAX_DURATION
            dur   = end - start

        if dur < MIN_DURATION:
            continue

        zones.append({
            "start":   round(start, 2),
            "end":     round(end, 2),
            "score":   round(zone_score, 1),
            "text":    zone_text,
            "reasons": reasons,
        })

    return zones


# ── Fallback: evenly-spaced clips when scoring yields too few zones ───────────
def _generate_fallback_zones(video_duration: float, n: int) -> list:
    """
    Space n clips evenly across the timeline.
    Used when scoring + dedup can't fill the requested clip count.
    """
    clip_dur = min(MIN_DURATION, MAX_DURATION)
    spacing  = video_duration / (n + 1)
    zones    = []
    for i in range(1, n + 1):
        mid   = i * spacing
        start = max(0.0, round(mid - clip_dur / 2, 2))
        end   = round(min(video_duration, start + clip_dur), 2)
        if end - start < 5:          # skip degenerate clips near end of video
            continue
        zones.append({
            "start":   start,
            "end":     end,
            "score":   0.0,
            "text":    "",
            "reasons": ["fallback-evenly-spaced"],
        })
    return zones


# ── Deduplication (remove overlapping zones, keep highest score) ──────────────
def dedup_zones(zones):
    zones = sorted(zones, key=lambda x: x["score"], reverse=True)
    kept = []
    for zone in zones:
        overlap = False
        for k in kept:
            # Overlapping if they share more than 50% of the shorter zone
            overlap_start = max(zone["start"], k["start"])
            overlap_end   = min(zone["end"],   k["end"])
            overlap_dur   = max(0, overlap_end - overlap_start)
            shorter_dur   = min(zone["end"] - zone["start"], k["end"] - k["start"])
            if shorter_dur > 0 and overlap_dur / shorter_dur > 0.5:
                overlap = True
                break
        if not overlap:
            kept.append(zone)

    return sorted(kept, key=lambda x: x["start"])


# ── Clip extraction ───────────────────────────────────────────────────────────
def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def collect_words_for_zone(segments, zone_start: float, zone_end: float):
    """Return word timestamps adjusted to be relative to the clip start."""
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            # Strictly exclude words entirely outside the zone
            if w.end <= zone_start or w.start >= zone_end:
                continue
            rel_start = max(0.0, round(w.start - zone_start, 3))
            rel_end   = round(min(w.end, zone_end) - zone_start, 3)
            if rel_start >= rel_end:
                continue
            words.append({
                "word":  censor(w.word.strip()),
                "start": rel_start,
                "end":   rel_end,
            })
    return words


_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",          # macOS
    "/Library/Fonts/Arial Bold.ttf",                # macOS
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",           # Linux
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",            # Linux
]


def _add_thumbnail_overlay(image_path: str, title: str) -> None:
    """Burn a dark gradient + bold title text onto the thumbnail in-place."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # Gradient: transparent at 52% height, opaque black at bottom
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gdraw   = ImageDraw.Draw(overlay)
    g_start = int(h * 0.52)
    for y in range(g_start, h):
        t     = (y - g_start) / (h - g_start)
        alpha = int(220 * (t ** 0.65))
        gdraw.rectangle([0, y, w, y + 1], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, overlay)

    # Font — try bold variants first
    font_size = 76
    font      = None
    for path in _FONT_PATHS:
        if not os.path.exists(path):
            continue
        for idx in (1, 0):      # index 1 = bold face in .ttc collections
            try:
                font = ImageFont.truetype(path, font_size, index=idx)
                break
            except Exception:
                continue
        if font:
            break
    if font is None:
        img.convert("RGB").save(image_path, "JPEG", quality=92)
        return

    # Word-wrap to max 3 lines, each ≤ 85% image width
    words   = title.split()
    lines, line = [], []
    for word in words:
        test = " ".join(line + [word])
        if font.getlength(test) > w * 0.85 and line:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    lines = lines[:3]

    draw   = ImageDraw.Draw(img)
    line_h = font_size + 18
    y0     = h - len(lines) * line_h - 100

    for line_text in lines:
        tw = font.getlength(line_text)
        x  = int((w - tw) / 2)
        draw.text((x, y0), line_text, font=font,
                  fill=(255, 255, 255, 255),
                  stroke_width=4, stroke_fill=(0, 0, 0, 255))
        y0 += line_h

    img.convert("RGB").save(image_path, "JPEG", quality=92)


def generate_thumbnail(clip_path: str, duration: float, title: str = ""):
    """
    Extract a 9:16 frame from the clip, apply blurred-background compositing
    + colour enhancement, then burn in a gradient + title overlay with Pillow.
    """
    thumb_path = os.path.splitext(clip_path)[0] + "_thumb.jpg"
    seek = max(0.5, min(duration * 0.25, duration - 1.0))

    filter_complex = (
        "split[a][b];"
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=25[bg];"
        "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        "eq=saturation=1.35:contrast=1.1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek), "-i", clip_path,
        "-filter_complex", filter_complex,
        "-vframes", "1", "-update", "1", "-q:v", "2",
        thumb_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0 or not os.path.exists(thumb_path):
            err = result.stderr.decode(errors="replace")[-300:]
            print(f"    thumbnail failed (rc={result.returncode}): {err}", flush=True)
            return None
    except subprocess.TimeoutExpired:
        print("    thumbnail timed out", flush=True)
        return None
    except FileNotFoundError:
        return None

    if title:
        _add_thumbnail_overlay(thumb_path, title)

    print("    📸 thumbnail generated", flush=True)
    return thumb_path


def _run_ffmpeg(cmd, timeout=300):
    """Run ffmpeg, return True on success."""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        print("  ✗ ffmpeg not found — install ffmpeg and make sure it is on your PATH", flush=True)
        sys.exit(1)


def _ts_ass(seconds: float) -> str:
    """Format seconds as h:mm:ss.cc (ASS subtitle timestamp)."""
    total_cs = round(max(0.0, seconds) * 100)
    h        = total_cs // 360000; total_cs %= 360000
    m        = total_cs // 6000;   total_cs %= 6000
    s        = total_cs // 100;    cs = total_cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _chunk_words(words, size=5, gap_thresh=1.5):
    """Split word list into caption chunks: new chunk on gap > gap_thresh or max size words."""
    chunks, cur = [], []
    for i, w in enumerate(words):
        gap = (w["start"] - words[i - 1]["end"]) if i > 0 else 0
        if cur and (len(cur) >= size or gap > gap_thresh):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _write_ass_subtitles(words: list, path: str) -> None:
    """Write an ASS subtitle file with karaoke word-level highlighting."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # PrimaryColour = gold (&H0000D7FF in BGR), SecondaryColour = white
        "Style: Default,Arial,52,&H0000D7FF,&H00FFFFFF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,3,1,2,20,20,60,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    chunks = _chunk_words(words)
    lines  = []
    for chunk in chunks:
        if not chunk:
            continue
        ev_start = chunk[0]["start"]
        ev_end   = chunk[-1]["end"]
        # Build karaoke text: {\k<cs>}word  (cs = duration of this word in centiseconds)
        karaoke_parts = []
        for w in chunk:
            dur_cs = max(1, round((w["end"] - w["start"]) * 100))
            karaoke_parts.append(f"{{\\k{dur_cs}}}{w['word']}")
        text = " ".join(karaoke_parts)
        lines.append(
            f"Dialogue: 0,{_ts_ass(ev_start)},{_ts_ass(ev_end)},Default,,0,0,0,,{text}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def _stream_copy_clip(input_file: str, start: float, duration: float, out_path: str) -> bool:
    """Fast stream copy with re-encode fallback."""
    ok = _run_ffmpeg([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", input_file,
        "-t", str(duration),
        "-c", "copy",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ], timeout=60)
    if not ok:
        ok = _run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", str(start), "-i", input_file,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ])
    return ok


def _burn_captions_watermark(input_file: str, start: float, duration: float,
                              words: list, out_path: str,
                              header_title: str = "") -> bool:
    """Re-encode clip with ASS captions, faint watermark, and optional punchy header."""
    import tempfile
    ass_fd, ass_path = tempfile.mkstemp(suffix=".ass")
    os.close(ass_fd)
    try:
        _write_ass_subtitles(words, ass_path)
        safe_ass = ass_path.replace("\\", "\\\\").replace(":", "\\:")

        vf_parts = [
            f"ass='{safe_ass}'",
            # Faint watermark — bottom-right corner
            "drawtext=text='Snipflow':fontcolor=white@0.22:fontsize=28:"
            "x=w-tw-20:y=h-th-20:shadowcolor=black@0.15:shadowx=1:shadowy=1",
        ]

        # Punchy header overlay — visible only during first HEADER_DURATION seconds
        if HEADER_ENABLED and header_title:
            safe_hdr = header_title.replace("'", "\\'").replace(":", "\\:")
            y_expr   = "80" if HEADER_POSITION == "top" else "(h-th)/2"
            vf_parts.append(
                f"drawtext=text='{safe_hdr}':fontcolor=white:fontsize=54:"
                f"x=(w-tw)/2:y={y_expr}:"
                f"shadowcolor=black@0.65:shadowx=3:shadowy=3:"
                f"enable='lt(t\\,{HEADER_DURATION})'"
            )

        return _run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", str(start), "-i", input_file,
            "-t", str(duration),
            "-vf", ",".join(vf_parts),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ], timeout=600)
    finally:
        try:
            os.unlink(ass_path)
        except OSError:
            pass


def _extract_one(i, zone, input_file, ext, video, annotate, segments):
    """Extract a single clip and its thumbnail. Designed to run in a thread pool."""
    start    = zone["start"]
    end      = zone["end"]
    duration = end - start
    score_i  = int(round(zone["score"]))
    title    = generate_clip_title(zone["text"], score=zone["score"])
    filename = f"clip_{i}_{safe_ts(start)}-{safe_ts(end)}_score_{score_i}.{ext}"
    out_path = os.path.join(CLIPS_DIR, filename)

    ok = False
    if video:
        # Collect words first so we can attempt caption burn-in
        words = collect_words_for_zone(segments, start, end) if (annotate and segments) else []

        if words and not FAST_MODE:
            ok = _burn_captions_watermark(
                input_file, start, duration, words, out_path,
                header_title=_thumbnail_label(title, score=zone["score"]),
            )
            if not ok:
                print(f"  clip_{i} caption burn failed, falling back to stream copy...", flush=True)

        if not ok:
            ok = _stream_copy_clip(input_file, start, duration, out_path)
    else:
        words = collect_words_for_zone(segments, start, end) if (annotate and segments) else []
        ok = _run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", str(start), "-i", input_file,
            "-t", str(duration),
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            out_path,
        ])

    if not ok:
        print(f"  ✗ clip_{i} failed", flush=True)
        return None

    print(f"  ✓ clip_{i} [{fmt_ts(start)}-{fmt_ts(end)}] {duration:.0f}s — {title}", flush=True)

    clip = {
        "title":   title,
        "summary": generate_clip_summary(zone["text"], score=zone["score"]),
        "start":   start,
        "end":     end,
        "score":   zone["score"],
        "url":     f"/clips/{filename}",
        "text":    zone["text"],
        "words":   words,
    }

    if video:
        thumb_path = generate_thumbnail(
            out_path, duration,
            title=_thumbnail_label(title, score=zone["score"]),
        )
        if thumb_path:
            clip["thumbnail"] = f"/clips/{os.path.basename(thumb_path)}"

    return clip


def extract_clips(input_file: str, zones, annotate: bool = False, segments=None):
    os.makedirs(CLIPS_DIR, exist_ok=True)
    video   = is_video(input_file)
    ext     = "mp4" if video else "mp3"
    workers = min(os.cpu_count() or 2, len(zones), 2)   # cap at 2 to avoid I/O thrash

    print(f"\n🎬 Extracting {len(zones)} clip(s) — {workers} worker(s)"
          f" [FAST_MODE={'on' if FAST_MODE else 'off'}]...\n", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_extract_one, i, zone, input_file, ext, video, annotate, segments): i
            for i, zone in enumerate(zones, 1)
        }
        for fut in as_completed(futs):
            clip = fut.result()
            if clip:
                results.append(clip)

    # Restore chronological order (futures complete in arbitrary order)
    return sorted(results, key=lambda c: c["start"])


# ── Annotate-only mode ────────────────────────────────────────────────────────
def process_annotate_only(input_file: str) -> dict:
    """
    Transcribe the entire video with word timestamps and return it as a single
    annotated clip (no highlight detection, no cutting).
    """
    os.makedirs(CLIPS_DIR, exist_ok=True)
    audio_path      = None
    transcribe_path = input_file

    if is_video(input_file):
        print("Extracting audio...", flush=True)
        audio_path      = extract_audio_for_transcription(input_file)
        transcribe_path = audio_path

    try:
        segments = transcribe(transcribe_path, word_timestamps=True)
        print(f"  {len(segments)} segments transcribed", flush=True)

        if not segments:
            print("WARNING: no speech detected in video", file=sys.stderr)
            return {"clips": []}

        ext      = os.path.splitext(input_file)[1] or ".mp4"
        duration = segments[-1].end
        filename = f"annotated_{safe_ts(0)}-{safe_ts(duration)}.mp4"
        out_path = os.path.join(CLIPS_DIR, filename)

        # Collect all word timestamps (absolute, not offset)
        all_words = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                all_words.append({
                    "word":  censor(w.word.strip()),
                    "start": round(w.start, 3),
                    "end":   round(w.end,   3),
                })

        full_text = censor(clean(" ".join(seg.text for seg in segments)))
        title     = generate_clip_title(full_text)

        # Attempt caption burn-in + watermark; fall back to plain copy
        burned = False
        if all_words and is_video(input_file):
            burned = _burn_captions_watermark(
                input_file, 0, duration, all_words, out_path,
                header_title=_thumbnail_label(title),
            )
            if burned:
                print(f"  ✓ captions burned into clips/{filename}", flush=True)
            else:
                print("  caption burn failed, falling back to copy...", flush=True)
        if not burned:
            shutil.copy2(input_file, out_path)
            print(f"  ✓ copied input to clips/{filename}", flush=True)

        thumb_path = generate_thumbnail(out_path, duration, title=_thumbnail_label(title))

        clip = {
            "title":   title,
            "summary": generate_clip_summary(full_text),
            "start":   0,
            "end":     round(duration, 2),
            "url":     f"/clips/{filename}",
            "text":    full_text,
            "words":   all_words,
        }
        if thumb_path:
            clip["thumbnail"] = f"/clips/{os.path.basename(thumb_path)}"

        return {"clips": [clip]}

    finally:
        if audio_path and audio_path != input_file and os.path.exists(audio_path):
            os.unlink(audio_path)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        input_file       = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE
        output_json_path = sys.argv[2] if len(sys.argv) > 2 else None
        mode_arg         = sys.argv[3] if len(sys.argv) > 3 else ""
        clip_length      = sys.argv[4] if len(sys.argv) > 4 else "medium"
        max_clips        = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        annotate_only    = mode_arg == "annotate_only"
        annotate         = mode_arg == "annotate"

        # Apply clip-length preset — overrides module-level duration constants
        preset = DURATION_PRESETS.get(clip_length, DURATION_PRESETS["medium"])
        TARGET_DURATION = preset["target"]
        MIN_DURATION    = preset["min_dur"]
        MAX_DURATION    = preset["max_dur"]
        CLUSTER_GAP     = preset["cluster_gap"]
        print(f"Clip-length preset: {clip_length} "
              f"(target={TARGET_DURATION}s, min={MIN_DURATION}s, max={MAX_DURATION}s)", flush=True)

        if not os.path.exists(input_file):
            print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)

        if annotate_only:
            output = process_annotate_only(input_file)
        else:
            # Extract 16 kHz mono WAV — Whisper's native format AND source for RMS energy scoring
            audio_path      = None
            transcribe_path = input_file
            if is_video(input_file):
                print("Extracting audio...", flush=True)
                audio_path      = extract_audio_for_transcription(input_file)
                transcribe_path = audio_path

            try:
                # Always use word_timestamps=True so every clip includes captions
                segments = transcribe(transcribe_path, word_timestamps=True)
                print(f"  {len(segments)} segments transcribed", flush=True)

                if not segments:
                    print("WARNING: no speech detected in video", file=sys.stderr)
                    output = {"clips": []}
                else:
                    # audio_path (WAV) stays alive here so find_clip_zones can read energy
                    zones = find_clip_zones(segments, audio_path=transcribe_path,
                                            max_clips=max_clips)
                    zones = dedup_zones(zones)

                    # ── Clip count logic ──────────────────────────────────────
                    if max_clips > 0:
                        if len(zones) < max_clips:
                            # Pass 1 failed — retry with no score floor
                            print(f"  ⚠ Only {len(zones)}/{max_clips} zones found; "
                                  "retrying with min_zone_score=0...", flush=True)
                            all_zones = find_clip_zones(
                                segments, audio_path=transcribe_path,
                                min_zone_score=0, max_clips=max_clips,
                            )
                            all_zones = dedup_zones(all_zones)
                            zones = sorted(all_zones, key=lambda z: z["score"],
                                           reverse=True)[:max_clips]
                        else:
                            zones = sorted(zones, key=lambda z: z["score"],
                                           reverse=True)[:max_clips]

                        # Pass 2 still not enough → fill with evenly-spaced fallback clips
                        if len(zones) < max_clips:
                            video_dur = segments[-1].end
                            needed    = max_clips - len(zones)
                            fb_zones  = _generate_fallback_zones(video_dur, needed)
                            # Exclude fallback zones that overlap existing scored zones
                            existing_ends = {(z["start"], z["end"]) for z in zones}
                            fb_zones = [
                                fz for fz in fb_zones
                                if not any(
                                    max(fz["start"], z["start"]) < min(fz["end"], z["end"])
                                    for z in zones
                                )
                            ]
                            zones = sorted(zones + fb_zones[:needed],
                                           key=lambda z: z["start"])
                            print(f"  ℹ Added {len(fb_zones[:needed])} evenly-spaced "
                                  "fallback clip(s)", flush=True)
                        else:
                            zones = sorted(zones, key=lambda z: z["start"])

                    print(f"\n✅ {len(zones)}/{max_clips if max_clips else 'auto'} "
                          f"clip(s) returned\n", flush=True)
                    for i, z in enumerate(zones, 1):
                        print(f"  {i}. [{fmt_ts(z['start'])}-{fmt_ts(z['end'])}] "
                              f"{z['end']-z['start']:.0f}s | score {z['score']}", flush=True)

                    # Always pass segments so every clip gets word timestamps
                    clips  = extract_clips(input_file, zones, annotate=True,
                                           segments=segments) if zones else []
                    output = {"clips": clips}

            finally:
                if audio_path and audio_path != input_file and os.path.exists(audio_path):
                    os.unlink(audio_path)

        if output_json_path:
            with open(output_json_path, "w") as f:
                json.dump(output, f)
        else:
            print(json.dumps(output))

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
