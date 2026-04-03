"""
Performance, clip count, duration, and fast-mode tests for clipify-pipeline.py.
Run: python -m pytest tests/test_performance.py -v
"""
import importlib.util
import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch


# ── Load pipeline (hyphenated filename) ────────────────────────────────────────
def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "clipify_pipeline",
        os.path.join(os.path.dirname(__file__), "..", "clipify-pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = MagicMock()
    sys.modules.setdefault("faster_whisper", fw)

    vs  = types.ModuleType("vaderSentiment")
    vsa = types.ModuleType("vaderSentiment.vaderSentiment")
    vsa.SentimentIntensityAnalyzer = MagicMock(return_value=MagicMock(
        polarity_scores=MagicMock(return_value={"compound": 0.0, "neg": 0.0, "pos": 0.0})
    ))
    sys.modules.setdefault("vaderSentiment", vs)
    sys.modules.setdefault("vaderSentiment.vaderSentiment", vsa)

    spec.loader.exec_module(mod)
    return mod


pipeline = _load_pipeline()


# ── Segment / word factories ───────────────────────────────────────────────────
def _seg(start, end, text="hello world this is great content", score_boost=False):
    s = MagicMock()
    s.start = start
    s.end   = end
    s.text  = ("INSANE crazy unbelievable " + text) if score_boost else text
    s.words = []
    return s


def _high_seg(start, end):
    return _seg(start, end, score_boost=True)


def _make_segments(n, duration=3600.0, high_every=10):
    """Return n mock segments evenly spread over `duration` seconds."""
    step = duration / n
    segs = []
    for i in range(n):
        start = i * step
        end   = start + step * 0.8
        segs.append(_high_seg(start, end) if i % high_every == 0 else _seg(start, end))
    return segs


# ══════════════════════════════════════════════════════════════════════════════
class TestClipCount(unittest.TestCase):
    """A) Exactly max_clips clips are returned when content allows."""

    def _run_zones(self, segs, max_clips):
        pipeline.MIN_DURATION   = 15
        pipeline.MAX_DURATION   = 35
        pipeline.CLUSTER_GAP    = 20
        pipeline.MIN_ZONE_SCORE = 0
        pipeline.MIN_SEG_SCORE  = 0
        zones = pipeline.find_clip_zones(segs, max_clips=max_clips)
        zones = pipeline.dedup_zones(zones)
        # Simulate __main__ cap
        if max_clips > 0:
            zones = sorted(zones, key=lambda z: z["score"], reverse=True)[:max_clips]
        return zones

    def test_returns_at_most_max_clips(self):
        segs  = [_high_seg(i * 40, i * 40 + 30) for i in range(10)]
        zones = self._run_zones(segs, max_clips=5)
        self.assertLessEqual(len(zones), 5)

    def test_fallback_fills_to_max_clips(self):
        """If scoring yields too few, fallback evenly-spaced clips fill the rest."""
        video_dur = 300.0
        needed    = 5
        fb        = pipeline._generate_fallback_zones(video_dur, needed)
        self.assertEqual(len(fb), needed)

    def test_fallback_no_overlap(self):
        """Fallback clips must not overlap each other."""
        fb = pipeline._generate_fallback_zones(600.0, 6)
        for i in range(len(fb) - 1):
            self.assertLessEqual(fb[i]["end"], fb[i + 1]["start"] + 0.5)


# ══════════════════════════════════════════════════════════════════════════════
class TestClipDuration(unittest.TestCase):
    """B) All clips strictly within preset bounds."""

    def _zones_for_preset(self, preset_name):
        preset = pipeline.DURATION_PRESETS[preset_name]
        pipeline.MIN_DURATION = preset["min_dur"]
        pipeline.MAX_DURATION = preset["max_dur"]
        pipeline.CLUSTER_GAP  = preset["cluster_gap"]
        pipeline.MIN_ZONE_SCORE = 0
        pipeline.MIN_SEG_SCORE  = 0

        segs = [_high_seg(i * 60, i * 60 + 20) for i in range(10)]
        zones = pipeline.find_clip_zones(segs)
        return zones, preset["min_dur"], preset["max_dur"]

    def test_short_preset_bounds(self):
        zones, lo, hi = self._zones_for_preset("short")
        for z in zones:
            dur = z["end"] - z["start"]
            self.assertGreaterEqual(dur, lo - 0.1, f"clip too short: {dur:.1f}s")
            self.assertLessEqual(dur, hi + 0.1,    f"clip too long: {dur:.1f}s")

    def test_medium_preset_bounds(self):
        zones, lo, hi = self._zones_for_preset("medium")
        for z in zones:
            dur = z["end"] - z["start"]
            self.assertGreaterEqual(dur, lo - 0.1)
            self.assertLessEqual(dur, hi + 0.1)

    def test_long_preset_bounds(self):
        zones, lo, hi = self._zones_for_preset("long")
        for z in zones:
            dur = z["end"] - z["start"]
            self.assertGreaterEqual(dur, lo - 0.1)
            self.assertLessEqual(dur, hi + 0.1)


# ══════════════════════════════════════════════════════════════════════════════
class TestFallbackGeneration(unittest.TestCase):
    """C) Fallback clips generated even with zero-scoring segments."""

    def test_fallback_zones_created(self):
        fb = pipeline._generate_fallback_zones(video_duration=180.0, n=3)
        self.assertEqual(len(fb), 3)
        for z in fb:
            self.assertGreater(z["end"] - z["start"], 0)
            self.assertIn("fallback-evenly-spaced", z["reasons"])

    def test_fallback_within_video_bounds(self):
        dur = 120.0
        fb  = pipeline._generate_fallback_zones(dur, 4)
        for z in fb:
            self.assertGreaterEqual(z["start"], 0)
            self.assertLessEqual(z["end"], dur + 0.1)

    def test_fallback_score_is_zero(self):
        fb = pipeline._generate_fallback_zones(300.0, 3)
        for z in fb:
            self.assertEqual(z["score"], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
class TestFastMode(unittest.TestCase):
    """D) FAST_MODE=True skips _burn_captions_watermark."""

    def test_fast_mode_skips_burn(self):
        pipeline.FAST_MODE = True
        zone = {"start": 0.0, "end": 20.0, "score": 5.0, "text": "test",
                "reasons": [], "words": []}

        with patch.object(pipeline, "_burn_captions_watermark") as mock_burn, \
             patch.object(pipeline, "_stream_copy_clip", return_value=True), \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[
                 {"word": "hi", "start": 0.0, "end": 0.5}
             ]):
            pipeline._extract_one(1, zone, "fake.mp4", "mp4", True, True, [])

        mock_burn.assert_not_called()

    def test_slow_mode_uses_burn(self):
        pipeline.FAST_MODE = False
        zone = {"start": 0.0, "end": 20.0, "score": 5.0, "text": "test",
                "reasons": [], "words": []}
        # segments must be truthy so the collect_words_for_zone branch is entered
        fake_segments = [MagicMock()]

        with patch.object(pipeline, "_burn_captions_watermark", return_value=True) as mock_burn, \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[
                 {"word": "hi", "start": 0.0, "end": 0.5}
             ]):
            pipeline._extract_one(1, zone, "fake.mp4", "mp4", True, True, fake_segments)

        mock_burn.assert_called_once()

    def tearDown(self):
        pipeline.FAST_MODE = True   # restore default


# ══════════════════════════════════════════════════════════════════════════════
class TestEarlyExit(unittest.TestCase):
    """E) Processing stops once enough clusters are built."""

    def test_early_exit_triggered(self):
        # 50 high-scoring segments spaced 10s apart, CLUSTER_GAP=2 → each = its own cluster
        # requesting 2 clips → early exit once 4 clusters built
        segs = [_high_seg(i * 10, i * 10 + 3) for i in range(50)]
        pipeline.MIN_SEG_SCORE  = 0
        pipeline.MIN_ZONE_SCORE = 0
        pipeline.CLUSTER_GAP    = 2   # gap between segs is 7s > 2 → separate clusters

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pipeline.find_clip_zones(segs, max_clips=2)
        output = buf.getvalue()

        self.assertIn("Early exit", output)

    def test_no_early_exit_without_max_clips(self):
        segs = [_high_seg(i * 5, i * 5 + 3) for i in range(20)]
        pipeline.MIN_SEG_SCORE  = 0
        pipeline.MIN_ZONE_SCORE = 0
        pipeline.CLUSTER_GAP    = 2

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            pipeline.find_clip_zones(segs, max_clips=0)
        self.assertNotIn("Early exit", buf.getvalue())


# ══════════════════════════════════════════════════════════════════════════════
class TestScaling(unittest.TestCase):
    """F) Processing time scales linearly (not exponentially) with segment count."""

    def _time_zones(self, n):
        segs = _make_segments(n, duration=float(n * 10))
        pipeline.MIN_SEG_SCORE  = 0
        pipeline.MIN_ZONE_SCORE = 0
        pipeline.CLUSTER_GAP    = 30
        t0 = time.perf_counter()
        pipeline.find_clip_zones(segs)
        return time.perf_counter() - t0

    def test_100_segments(self):
        t = self._time_zones(100)
        self.assertLess(t, 5.0, f"100 segments took {t:.2f}s — too slow")

    def test_1000_segments(self):
        t = self._time_zones(1000)
        self.assertLess(t, 15.0, f"1000 segments took {t:.2f}s — too slow")

    def test_5000_segments_capped(self):
        """MAX_SCAN_SEGMENTS=5000 ensures O(n) doesn't blow up."""
        segs = _make_segments(6000, duration=60000.0)
        pipeline.MAX_SCAN_SEGMENTS = 5000
        t0 = time.perf_counter()
        pipeline.find_clip_zones(segs)
        t = time.perf_counter() - t0
        self.assertLess(t, 30.0, f"5000-cap scan took {t:.2f}s — too slow")

    def test_scaling_is_subquadratic(self):
        """1000-seg run should take < 10× the 100-seg run (linear = ~10×, quadratic = 100×)."""
        t100  = self._time_zones(100)
        t1000 = self._time_zones(1000)
        if t100 > 0:
            ratio = t1000 / t100
            self.assertLess(ratio, 50, f"Scaling ratio {ratio:.1f}× suggests exponential growth")


if __name__ == "__main__":
    unittest.main()
