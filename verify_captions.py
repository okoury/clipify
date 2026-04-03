#!/usr/bin/env python3
"""
Manual verification script for caption burn-in and watermark.
Usage: python verify_captions.py <input_video.mp4>
"""
import os
import sys
import tempfile
import importlib.util
import types
from unittest.mock import MagicMock


def load_pipeline():
    import faster_whisper  # noqa — must be installed
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa
    spec = importlib.util.spec_from_file_location(
        "clipify_pipeline",
        os.path.join(os.path.dirname(__file__), "clipify-pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_captions.py <input_video.mp4>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    print("Loading pipeline...")
    p = load_pipeline()

    # ── Test 1: _ts_ass correctness ─────────────────────────────────────────
    print("\n[1] _ts_ass checks")
    cases = [(0.0, "0:00:00.00"), (1.0, "0:00:01.00"), (3723.45, "1:02:03.45")]
    for secs, expected in cases:
        got = p._ts_ass(secs)
        status = "✓" if got == expected else f"✗ expected {expected}"
        print(f"  {secs}s → {got}  {status}")

    # ── Test 2: _chunk_words gap detection ──────────────────────────────────
    print("\n[2] _chunk_words gap detection")
    words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
        {"word": "gap",   "start": 3.0, "end": 3.4},  # 2.1s gap
        {"word": "here",  "start": 3.5, "end": 3.9},
    ]
    chunks = p._chunk_words(words)
    print(f"  Input: 4 words with 2.1s gap mid-way")
    print(f"  Chunks: {len(chunks)} (expected 2)")
    print(f"  Chunk 1: {[w['word'] for w in chunks[0]]}")
    print(f"  Chunk 2: {[w['word'] for w in chunks[1]]}")

    # ── Test 3: collect_words_for_zone boundary clamping ────────────────────
    print("\n[3] collect_words_for_zone boundary clamping")
    w1 = MagicMock(); w1.word = " early"; w1.start = 4.8; w1.end = 5.2
    w2 = MagicMock(); w2.word = " inside"; w2.start = 6.0; w2.end = 6.5
    w3 = MagicMock(); w3.word = " late"; w3.start = 9.8; w3.end = 10.3
    seg = MagicMock(); seg.words = [w1, w2, w3]
    result = p.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
    for r in result:
        neg = "✗ NEGATIVE" if r["start"] < 0 else "✓"
        print(f"  '{r['word']}' start={r['start']} end={r['end']}  {neg}")

    # ── Test 4: Write ASS file and inspect ──────────────────────────────────
    print("\n[4] ASS subtitle file generation")
    test_words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.1},
        {"word": "this",  "start": 1.2, "end": 1.5},
    ]
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
        ass_path = f.name
    try:
        p._write_ass_subtitles(test_words, ass_path)
        content = open(ass_path, encoding="utf-8").read()
        print(f"  File written: {ass_path}")
        print(f"  Has [Script Info]: {'✓' if '[Script Info]' in content else '✗'}")
        print(f"  Has [Events]:      {'✓' if '[Events]' in content else '✗'}")
        print(f"  Has karaoke tags:  {'✓' if '{\\k' in content else '✗'}")
        print(f"  Has gold colour:   {'✓' if '0000D7FF' in content else '✗'}")
        print("\n  First Dialogue line:")
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                print(f"    {line[:120]}")
                break
    finally:
        os.unlink(ass_path)

    # ── Test 5: Full burn-in on real video ──────────────────────────────────
    print(f"\n[5] Full caption burn-in on: {input_file}")
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    try:
        words_real = [
            {"word": "Test",    "start": 0.0, "end": 0.5},
            {"word": "caption", "start": 0.6, "end": 1.1},
            {"word": "burn-in", "start": 1.2, "end": 1.7},
            {"word": "and",     "start": 1.8, "end": 2.0},
            {"word": "Snipflow","start": 2.1, "end": 2.6},
        ]
        import subprocess
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_file],
            capture_output=True, text=True
        )
        duration = 10.0  # default
        if probe.returncode == 0:
            import json
            info = json.loads(probe.stdout)
            duration = min(float(info["format"].get("duration", 10)), 10)

        ok = p._burn_captions_watermark(input_file, 0, duration, words_real, out_path)
        if ok:
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  ✓ Output: {out_path} ({size_mb:.1f} MB)")
            print(f"  Open in a video player to verify:")
            print(f"    - Captions appear bottom-center with word highlighting")
            print(f"    - 'Snipflow' watermark visible bottom-right (faint)")
        else:
            print("  ✗ Burn-in failed — check ffmpeg has libass support:")
            print("    ffmpeg -filters | grep ass")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        if os.path.exists(out_path):
            os.unlink(out_path)

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
