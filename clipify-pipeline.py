from faster_whisper import WhisperModel
import re
import subprocess
import sys
import json
import os

AUDIO_FILE = "audio.mp3"
CLIPS_DIR = "clips"
MODEL_SIZE = "base"
GROUP_SIZE = 3
TOP_N = 5
OUTPUT_FILE = "top_chunks.txt"
TIMESTAMPS_FILE = "clip_timestamps.txt"
CLIPS_TO_EXPORT = 3
CLIP_PADDING = 1.5
MIN_CLIP_DURATION = 5
MERGE_MAX_GAP = 3


def transcribe_segments(path: str):
    print("Loading model...")
    model = WhisperModel(MODEL_SIZE)

    print(f"Transcribing {path}...")
    segments, _ = model.transcribe(path)

    return list(segments)


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def safe_timestamp_for_filename(seconds: float) -> str:
    return format_timestamp(seconds).replace(":", "-")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def trim_to_sentence(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences[0] if sentences else text


def group_segments(segments, group_size: int = GROUP_SIZE):
    grouped = []

    for i in range(0, len(segments), group_size):
        batch = segments[i:i + group_size]
        if not batch:
            continue

        text = " ".join(segment.text.strip() for segment in batch)
        text = clean_text(text)

        if not text:
            continue

        grouped.append({
            "start": batch[0].start,
            "end": batch[-1].end,
            "text": text
        })

    return grouped


def score_chunk(chunk_text: str):
    chunk_lower = chunk_text.lower()
    score = 0
    reasons = []

    keywords = [
        "crazy", "insane", "amazing", "unbelievable",
        "wow", "no way", "wtf", "what", "why",
        "huge", "massive", "problem", "mistake",
        "secret", "truth", "actually", "literally",
        "exciting", "interesting", "important",
        "wild", "bro", "nah", "impossible", "ridiculous"
    ]

    hook_words = [
        "how", "why", "what happened", "wait",
        "listen", "look", "here's", "this is why",
        "the reason", "what if"
    ]

    structure_words = [
        "first", "second", "third", "top",
        "best", "worst"
    ]

    for word in keywords:
        if word in chunk_lower:
            score += 2
            reasons.append(f'keyword: "{word}"')

    for hook in hook_words:
        if hook in chunk_lower:
            score += 2
            reasons.append(f'hook: "{hook}"')

    for word in structure_words:
        if word in chunk_lower:
            score += 2
            reasons.append(f'structure: "{word}"')

    exclamations = chunk_text.count("!")
    questions = chunk_text.count("?")

    if exclamations:
        score += exclamations * 3
        reasons.append(f"exclamation marks: {exclamations}")

    if questions:
        score += questions * 2
        reasons.append(f"question marks: {questions}")

    word_count = len(chunk_text.split())
    if 20 <= word_count <= 80:
        score += 2
        reasons.append("good clip length")
    elif word_count < 8:
        score -= 2
        reasons.append("too short")

    if re.search(r"\b(really|very|so|extremely)\b", chunk_lower):
        score += 1
        reasons.append("emphasis word")

    return max(score, 0), reasons


def build_scored_chunks(chunks):
    scored_chunks = []

    for chunk in chunks:
        score, reasons = score_chunk(chunk["text"])

        if score <= 0:
            continue

        scored_chunks.append({
            "score": score,
            "reasons": reasons,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        })

    return scored_chunks


def normalize_text_for_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def remove_near_duplicates(scored_chunks, min_gap_seconds: int = 20):
    filtered = []

    for chunk in scored_chunks:
        keep = True
        chunk_text_normalized = normalize_text_for_compare(chunk["text"])

        for existing in filtered:
            existing_text_normalized = normalize_text_for_compare(existing["text"])

            starts_too_close = abs(chunk["start"] - existing["start"]) < min_gap_seconds
            text_overlap = (
                chunk_text_normalized in existing_text_normalized
                or existing_text_normalized in chunk_text_normalized
            )

            if starts_too_close or text_overlap:
                keep = False
                break

        if keep:
            filtered.append(chunk)

    return filtered


def merge_close_chunks(scored_chunks, max_gap: int = MERGE_MAX_GAP):
    if not scored_chunks:
        return []

    sorted_chunks = sorted(scored_chunks, key=lambda x: x["start"])
    merged = [sorted_chunks[0].copy()]

    for chunk in sorted_chunks[1:]:
        last = merged[-1]

        if chunk["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], chunk["end"])
            combined_text = clean_text(last["text"] + " " + chunk["text"])
            last["text"] = trim_to_sentence(combined_text)
            last["score"] += chunk["score"]
            last["reasons"] = list(dict.fromkeys(last["reasons"] + chunk["reasons"]))
        else:
            merged.append(chunk.copy())

    return merged


def print_and_save_top_chunks(
    scored_chunks,
    top_n: int = TOP_N,
    output_file: str = OUTPUT_FILE,
    timestamps_file: str = TIMESTAMPS_FILE
):
    print("\n🔥 Top interesting chunks:\n")

    with open(output_file, "w", encoding="utf-8") as f, open(timestamps_file, "w", encoding="utf-8") as tf:
        for i, chunk in enumerate(scored_chunks[:top_n], start=1):
            start_label = format_timestamp(chunk["start"])
            end_label = format_timestamp(chunk["end"])
            reasons_text = ", ".join(chunk["reasons"]) if chunk["reasons"] else "none"

            output = (
                f"Rank: {i}\n"
                f"Score: {chunk['score']}\n"
                f"Timestamp: {start_label} - {end_label}\n"
                f"Reasons: {reasons_text}\n"
                f"{chunk['text']}\n\n"
            )

            print(output)
            f.write(output)
            tf.write(f"{start_label} - {end_label}\n")

    print(f"Saved results to {output_file}")
    print(f"Saved timestamps to {timestamps_file}")


def extract_audio_clips(input_file: str, scored_chunks, top_n: int = CLIPS_TO_EXPORT):
    print("\n🎬 Extracting audio clips...\n")
    os.makedirs(CLIPS_DIR, exist_ok=True)
    results = []

    for i, chunk in enumerate(scored_chunks[:top_n], start=1):
        start = max(chunk["start"] - CLIP_PADDING, 0)
        end = chunk["end"] + CLIP_PADDING

        if (end - start) < MIN_CLIP_DURATION:
            print(f"Skipping clip_{i} because duration is too short.")
            continue

        start_label = safe_timestamp_for_filename(start)
        end_label = safe_timestamp_for_filename(end)
        filename = f"clip_{i}_{start_label}-{end_label}_score_{chunk['score']}.mp3"
        output_file = os.path.join(CLIPS_DIR, filename)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_file,
                "-ss", str(start),
                "-to", str(end),
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "2",
                output_file
            ],
            check=True
        )

        print(f"Created {output_file} ({format_timestamp(start)} - {format_timestamp(end)})")

        results.append({
            "title": f"Clip {i}",
            "start": round(chunk["start"], 2),
            "end": round(chunk["end"], 2),
            "score": chunk["score"],
            "url": f"/clips/{filename}",
            "text": chunk["text"],
        })

    return results


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else AUDIO_FILE
    output_json_path = sys.argv[2] if len(sys.argv) > 2 else None

    segments = transcribe_segments(audio_file)
    chunks = group_segments(segments)

    scored_chunks = build_scored_chunks(chunks)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    scored_chunks = remove_near_duplicates(scored_chunks)
    scored_chunks = merge_close_chunks(scored_chunks)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    if not scored_chunks:
        print("No strong highlight chunks found. Try lowering the filter or changing keyword logic.")
        clips = []
    else:
        print_and_save_top_chunks(scored_chunks)
        clips = extract_audio_clips(audio_file, scored_chunks)

    output = {"clips": clips}

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(output, f)
    else:
        print(json.dumps(output))