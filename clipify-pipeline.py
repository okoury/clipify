from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import subprocess
import sys
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_FILE      = "audio.mp3"
CLIPS_DIR       = "clips"
MODEL_SIZE      = "small"    # good accuracy/speed tradeoff

MIN_SEG_SCORE   = 2          # min score for a segment to be "interesting"
CLUSTER_GAP     = 30         # seconds: high-scoring segs within this → one clip
TARGET_DURATION = 60         # ideal clip length (seconds) for TikTok/Shorts
MIN_DURATION    = 20         # discard clips shorter than this
MAX_DURATION    = 90         # cap clips at this (TikTok max = 60, Shorts = 60)
CLIP_PADDING    = 2.0        # seconds to extend before/after each clip zone
MIN_ZONE_SCORE  = 5          # minimum cumulative score for a zone to qualify

analyzer = SentimentIntensityAnalyzer()


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe(path: str):
    print("Loading Whisper model...")
    model = WhisperModel(MODEL_SIZE, compute_type="int8")
    print(f"Transcribing {path}...")
    segments, _ = model.transcribe(path, vad_filter=True)
    return list(segments)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_ts(s: float) -> str:
    t = int(s)
    return f"{t // 60:02d}:{t % 60:02d}"

def safe_ts(s: float) -> str:
    return fmt_ts(s).replace(":", "-")

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
]
STRUCTURE = [
    "first", "second", "third", "finally", "number one", "step",
    "tip", "rule", "most important", "the real reason", "top",
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

    if re.search(r"\b(really|very|so|extremely|absolutely|totally|completely)\b", lower):
        score += 1;  reasons.append("emphasis")

    caps = len(re.findall(r'\b[A-Z]{2,}\b', text))
    if caps:
        score += min(caps, 3);  reasons.append(f"{caps} caps word(s)")

    if words < 5:
        score -= 3

    return max(round(score, 1), 0.0), reasons


# ── Cluster high-scoring segments into clip zones ─────────────────────────────
def find_clip_zones(segments):
    """
    1. Score every segment individually.
    2. Collect segments that score >= MIN_SEG_SCORE.
    3. Cluster those segments: if two are within CLUSTER_GAP seconds, same clip.
    4. Expand each cluster toward TARGET_DURATION, cap at MAX_DURATION.
    5. Discard zones below MIN_ZONE_SCORE or shorter than MIN_DURATION.
    """
    scored_segs = []
    for seg in segments:
        text = clean(seg.text)
        score, reasons = score_segment(text, seg.end - seg.start)
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

    # Build clusters
    clusters = [[interesting[0]]]
    for seg in interesting[1:]:
        if seg["start"] - clusters[-1][-1]["end"] <= CLUSTER_GAP:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    zones = []
    for cluster in clusters:
        zone_start = cluster[0]["start"]
        zone_end   = cluster[-1]["end"]
        zone_dur   = zone_end - zone_start
        zone_score = sum(s["score"] for s in cluster)
        zone_text  = clean(" ".join(s["text"] for s in cluster))
        reasons    = []
        for s in cluster:
            reasons.extend(s["reasons"])
        reasons = list(dict.fromkeys(reasons))

        if zone_score < MIN_ZONE_SCORE:
            continue

        # Expand symmetrically toward TARGET_DURATION
        shortfall = max(0, TARGET_DURATION - zone_dur)
        expand    = shortfall / 2
        start     = max(0, zone_start - expand - CLIP_PADDING)
        end       = zone_end + expand + CLIP_PADDING

        # If still over MAX_DURATION, center on the highest-scoring segment
        if end - start > MAX_DURATION:
            peak  = max(cluster, key=lambda x: x["score"])
            mid   = (peak["start"] + peak["end"]) / 2
            start = max(0, mid - MAX_DURATION / 2)
            end   = start + MAX_DURATION

        if end - start < MIN_DURATION:
            continue

        zones.append({
            "start":   round(start, 2),
            "end":     round(end, 2),
            "score":   round(zone_score, 1),
            "text":    zone_text,
            "reasons": reasons,
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


def extract_clips(input_file: str, zones):
    print(f"\n🎬 Extracting {len(zones)} clips...\n")
    os.makedirs(CLIPS_DIR, exist_ok=True)
    video  = is_video(input_file)
    ext    = "mp4" if video else "mp3"
    results = []

    for i, zone in enumerate(zones, start=1):
        start    = zone["start"]
        end      = zone["end"]
        duration = end - start
        score_i  = int(round(zone["score"]))
        filename = f"clip_{i}_{safe_ts(start)}-{safe_ts(end)}_score_{score_i}.{ext}"
        out_path = os.path.join(CLIPS_DIR, filename)

        if video:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", input_file,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", input_file,
                "-t", str(duration),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                out_path,
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            dur_str = f"{duration:.0f}s"
            print(f"  ✓ clip_{i} [{fmt_ts(start)}-{fmt_ts(end)}] {dur_str} | score {zone['score']}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ clip_{i} failed: {e.stderr.decode()[:200] if e.stderr else 'unknown'}")
            continue

        results.append({
            "title": f"Clip {i}",
            "start": start,
            "end":   end,
            "score": zone["score"],
            "url":   f"/clips/{filename}",
            "text":  zone["text"],
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_file       = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else None

    segments = transcribe(input_file)
    print(f"  {len(segments)} segments transcribed")

    zones = find_clip_zones(segments)
    zones = dedup_zones(zones)

    print(f"\n✅ {len(zones)} highlight clip(s) found\n")
    for i, z in enumerate(zones, 1):
        print(f"  {i}. [{fmt_ts(z['start'])}-{fmt_ts(z['end'])}] "
              f"{z['end']-z['start']:.0f}s | score {z['score']}")

    clips  = extract_clips(input_file, zones) if zones else []
    output = {"clips": clips}

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(output, f)
    else:
        print(json.dumps(output))
