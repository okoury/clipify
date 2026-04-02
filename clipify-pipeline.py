from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import subprocess
import sys
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_FILE       = "audio.mp3"
CLIPS_DIR        = "clips"
MODEL_SIZE       = "small"   # better accuracy than "base"
GROUP_SIZE       = 2         # smaller = more granular detection
CLIP_PADDING     = 1.5       # seconds added before/after each clip
MIN_CLIP_DURATION = 4        # skip clips shorter than this
MERGE_MAX_GAP    = 2         # merge chunks within this many seconds
MIN_SCORE        = 3         # minimum score to qualify as a highlight
MIN_GAP_SECS     = 8         # min seconds between kept clips (dedup)

analyzer = SentimentIntensityAnalyzer()

# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_segments(path: str):
    print("Loading model...")
    model = WhisperModel(MODEL_SIZE, compute_type="int8")
    print(f"Transcribing {path}...")
    segments, _ = model.transcribe(path, vad_filter=True)
    return list(segments)


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"

def safe_ts(seconds: float) -> str:
    return format_timestamp(seconds).replace(":", "-")

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── Chunking (with overlapping window to catch boundary moments) ───────────────
def group_segments(segments, group_size: int = GROUP_SIZE):
    seg_list = list(segments)
    seen = set()
    grouped = []

    def make_chunk(batch):
        text = clean_text(" ".join(s.text.strip() for s in batch))
        if not text:
            return None
        key = (round(batch[0].start, 1), round(batch[-1].end, 1))
        if key in seen:
            return None
        seen.add(key)
        return {
            "start":    batch[0].start,
            "end":      batch[-1].end,
            "text":     text,
            "duration": batch[-1].end - batch[0].start,
        }

    # Non-overlapping pass
    for i in range(0, len(seg_list), group_size):
        chunk = make_chunk(seg_list[i:i + group_size])
        if chunk:
            grouped.append(chunk)

    # Overlapping pass (offset by 1) — catches moments at group boundaries
    for i in range(1, len(seg_list) - group_size + 1, group_size):
        chunk = make_chunk(seg_list[i:i + group_size])
        if chunk:
            grouped.append(chunk)

    return grouped


# ── Scoring ───────────────────────────────────────────────────────────────────
HIGH_IMPACT = [
    "crazy", "insane", "unbelievable", "incredible", "mind-blowing",
    "jaw-dropping", "shocking", "impossible", "never seen", "first time",
    "biggest ever", "worst ever", "greatest ever", "destroyed", "exposed",
    "secret", "nobody knows", "hidden truth", "revealed", "disaster",
    "crisis", "million dollars", "billion", "died", "survived", "broke",
    "scam", "fraud", "lie", "cheat", "betray", "attack", "arrested",
]

MEDIUM_IMPACT = [
    "amazing", "awesome", "wow", "huge", "massive", "problem", "mistake",
    "truth", "actually", "literally", "wild", "ridiculous", "serious",
    "important", "surprising", "unexpected", "won", "lost", "love", "hate",
    "angry", "scared", "excited", "money", "rich", "free", "no way", "wtf",
    "change", "different", "wrong", "right", "best", "worst", "fail",
]

HOOKS = [
    "wait", "listen", "watch this", "look at this", "check this out",
    "here's why", "the reason is", "what if", "turns out", "plot twist",
    "game changer", "i can't believe", "you won't believe", "let me show",
    "this is why", "spoiler", "breaking", "just happened", "right now",
    "oh my", "oh no", "oh wow", "you need to", "everyone needs to",
    "nobody talks about", "they don't want you to know",
]

STRUCTURE = [
    "first", "second", "third", "finally", "top", "number one", "step",
    "tip", "rule", "key point", "most important", "the real reason",
]


def score_chunk(text: str, duration: float):
    lower = text.lower()
    score = 0.0
    reasons = []

    # ── VADER sentiment (both high-positive AND high-negative are engaging) ──
    vader = analyzer.polarity_scores(text)
    compound = abs(vader["compound"])

    if compound >= 0.7:
        score += 6
        reasons.append(f"strong emotion ({compound:.2f})")
    elif compound >= 0.5:
        score += 4
        reasons.append(f"clear emotion ({compound:.2f})")
    elif compound >= 0.3:
        score += 2
        reasons.append(f"mild emotion ({compound:.2f})")

    if vader["neg"] > 0.25:
        score += 2
        reasons.append("tension/conflict")
    if vader["pos"] > 0.35:
        score += 2
        reasons.append("excitement/enthusiasm")

    # ── Keywords ──────────────────────────────────────────────────────────────
    for w in HIGH_IMPACT:
        if w in lower:
            score += 3
            reasons.append(f'"{w}"')

    for w in MEDIUM_IMPACT:
        if w in lower:
            score += 2
            reasons.append(f'"{w}"')

    for h in HOOKS:
        if h in lower:
            score += 3
            reasons.append(f'hook: "{h}"')

    for s in STRUCTURE:
        if s in lower:
            score += 1
            reasons.append(f'structure: "{s}"')

    # ── Punctuation ───────────────────────────────────────────────────────────
    exclamations = text.count("!")
    questions    = text.count("?")
    if exclamations:
        score += min(exclamations * 2, 6)
        reasons.append(f"{exclamations} exclamation(s)")
    if questions:
        score += min(questions * 1.5, 4)
        reasons.append(f"{questions} question(s)")

    # ── Speaking pace (fast = energetic/excited) ──────────────────────────────
    words = len(text.split())
    if duration > 0:
        pace = words / duration
        if pace > 3.0:
            score += 3
            reasons.append(f"fast pace ({pace:.1f} w/s)")
        elif pace > 2.2:
            score += 1.5
            reasons.append(f"active pace ({pace:.1f} w/s)")

    # ── Emphasis ──────────────────────────────────────────────────────────────
    if re.search(r"\b(really|very|so|extremely|absolutely|literally|totally|completely)\b", lower):
        score += 1
        reasons.append("emphasis")

    caps_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
    if caps_count:
        score += min(caps_count, 3)
        reasons.append(f"{caps_count} caps word(s)")

    # ── Penalty for very short chunks ─────────────────────────────────────────
    if words < 8:
        score -= 3

    return max(round(score, 1), 0), reasons


def build_scored_chunks(chunks):
    result = []
    for chunk in chunks:
        score, reasons = score_chunk(chunk["text"], chunk.get("duration", 5))
        if score >= MIN_SCORE:
            result.append({
                "score":   score,
                "reasons": reasons,
                "start":   chunk["start"],
                "end":     chunk["end"],
                "text":    chunk["text"],
            })
    return result


# ── Deduplication ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

def remove_near_duplicates(scored_chunks):
    # Sort by score so we always keep the highest-scored version of nearby chunks
    by_score = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)
    kept = []

    for chunk in by_score:
        norm = normalize(chunk["text"])
        discard = False

        for existing in kept:
            too_close   = abs(chunk["start"] - existing["start"]) < MIN_GAP_SECS
            text_overlap = norm in normalize(existing["text"]) or normalize(existing["text"]) in norm

            if too_close or text_overlap:
                discard = True
                break

        if not discard:
            kept.append(chunk)

    return kept


# ── Merging ───────────────────────────────────────────────────────────────────
def merge_close_chunks(scored_chunks):
    if not scored_chunks:
        return []

    by_time = sorted(scored_chunks, key=lambda x: x["start"])
    merged  = [by_time[0].copy()]

    for chunk in by_time[1:]:
        last = merged[-1]
        if chunk["start"] - last["end"] <= MERGE_MAX_GAP:
            last["end"]     = max(last["end"], chunk["end"])
            last["text"]    = clean_text(last["text"] + " " + chunk["text"])
            last["score"]   = max(last["score"], chunk["score"])
            last["reasons"] = list(dict.fromkeys(last["reasons"] + chunk["reasons"]))
        else:
            merged.append(chunk.copy())

    return merged


# ── Clip extraction ───────────────────────────────────────────────────────────
def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def extract_clips(input_file: str, scored_chunks):
    print(f"\n🎬 Extracting {len(scored_chunks)} clips...\n")
    os.makedirs(CLIPS_DIR, exist_ok=True)
    video = is_video(input_file)
    ext   = "mp4" if video else "mp3"
    results = []

    for i, chunk in enumerate(scored_chunks, start=1):
        start    = max(chunk["start"] - CLIP_PADDING, 0)
        end      = chunk["end"] + CLIP_PADDING
        duration = end - start

        if duration < MIN_CLIP_DURATION:
            print(f"  skip clip_{i}: too short ({duration:.1f}s)")
            continue

        score_int = int(round(chunk["score"]))
        filename  = f"clip_{i}_{safe_ts(start)}-{safe_ts(end)}_score_{score_int}.{ext}"
        out_path  = os.path.join(CLIPS_DIR, filename)

        if video:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", input_file,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", input_file,
                "-t", str(duration),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                out_path,
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ clip_{i}: {format_timestamp(start)} → {format_timestamp(end)} (score {chunk['score']})")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ clip_{i} failed: {e.stderr.decode()[:200]}")
            continue

        results.append({
            "title": f"Clip {i}",
            "start": round(chunk["start"], 2),
            "end":   round(chunk["end"], 2),
            "score": chunk["score"],
            "url":   f"/clips/{filename}",
            "text":  chunk["text"],
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_file       = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else None

    segments = transcribe_segments(input_file)
    chunks   = group_segments(segments)

    scored = build_scored_chunks(chunks)
    scored = remove_near_duplicates(scored)
    scored = merge_close_chunks(scored)
    scored.sort(key=lambda x: x["start"])  # chronological order for output

    print(f"\n✅ {len(scored)} highlight clips found\n")

    clips  = extract_clips(input_file, scored) if scored else []
    output = {"clips": clips}

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(output, f)
    else:
        print(json.dumps(output))
