"""
Integration tests: cut input.mp4 into 1–10 clips for every preset.

These tests use the REAL input.mp4 and real ffmpeg (no Whisper transcription).
They verify that get_video_duration() works, and that find_manual_clip_zones()
returns exactly the requested number of clips within preset duration bounds.

Run:  python -m pytest tests/test_integration.py -v
Skip: tests are skipped automatically when input.mp4 is absent.
"""
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

INPUT_MP4 = os.path.join(os.path.dirname(__file__), "..", "input.mp4")


# ── Load pipeline (stub out ML dependencies) ──────────────────────────────────
def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "clipify_pipeline_integ",
        os.path.join(os.path.dirname(__file__), "..", "clipify-pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = MagicMock()
    sys.modules.setdefault("faster_whisper", fw)

    vs  = types.ModuleType("vaderSentiment")
    vsa = types.ModuleType("vaderSentiment.vaderSentiment")
    vsa.SentimentIntensityAnalyzer = MagicMock(return_value=MagicMock(
        polarity_scores=MagicMock(
            return_value={"compound": 0.5, "neg": 0.1, "pos": 0.4, "neu": 0.5}
        )
    ))
    sys.modules.setdefault("vaderSentiment", vs)
    sys.modules.setdefault("vaderSentiment.vaderSentiment", vsa)

    spec.loader.exec_module(mod)
    return mod


pipeline = _load_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
@unittest.skipUnless(os.path.exists(INPUT_MP4), "input.mp4 not found — skipping integration tests")
class TestManualModeIntegration(unittest.TestCase):
    """
    Cuts input.mp4 into 1–10 clips using every preset.
    Asserts exact clip count and that every clip is within duration bounds.
    """

    @classmethod
    def setUpClass(cls):
        cls.duration = pipeline.get_video_duration(INPUT_MP4)
        print(f"\n  [integration] input.mp4 = {cls.duration:.2f}s  "
              f"({cls.duration / 60:.1f} min)", flush=True)
        assert cls.duration > 0, "get_video_duration returned 0 or negative"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_preset(self, name):
        preset = pipeline.DURATION_PRESETS[name]
        pipeline.TARGET_DURATION = preset["target"]
        pipeline.MIN_DURATION    = preset["min_dur"]
        pipeline.MAX_DURATION    = preset["max_dur"]
        return preset

    def _check_zones(self, zones, n, preset_name, preset):
        # 1. Exact count
        self.assertEqual(
            len(zones), n,
            f"[{preset_name}] n={n}: expected {n} zones, got {len(zones)}"
        )
        sorted_z = sorted(zones, key=lambda z: z["start"])
        for idx, z in enumerate(sorted_z):
            dur = z["end"] - z["start"]
            # 2. Duration within preset bounds (0.1 s tolerance for float rounding)
            self.assertGreaterEqual(
                dur, preset["min_dur"] - 0.1,
                f"[{preset_name}] n={n} zone[{idx}]: {dur:.2f}s < min {preset['min_dur']}s"
            )
            self.assertLessEqual(
                dur, preset["max_dur"] + 0.1,
                f"[{preset_name}] n={n} zone[{idx}]: {dur:.2f}s > max {preset['max_dur']}s"
            )
            # 3. Within video bounds
            self.assertGreaterEqual(z["start"], -0.01,
                f"[{preset_name}] n={n} zone[{idx}]: start {z['start']} < 0")
            self.assertLessEqual(
                z["end"], self.duration + 0.1,
                f"[{preset_name}] n={n} zone[{idx}]: end {z['end']} > video duration {self.duration}"
            )
        # 4. No overlaps
        for idx in range(len(sorted_z) - 1):
            self.assertLessEqual(
                sorted_z[idx]["end"], sorted_z[idx + 1]["start"] + 0.6,
                f"[{preset_name}] n={n}: zone[{idx}] and zone[{idx+1}] overlap"
            )

    # ── one test method per preset so failures are reported independently ─────

    def test_short_1_clip(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 1), 1, "short", p)

    def test_short_2_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 2), 2, "short", p)

    def test_short_3_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 3), 3, "short", p)

    def test_short_4_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 4), 4, "short", p)

    def test_short_5_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 5), 5, "short", p)

    def test_short_6_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 6), 6, "short", p)

    def test_short_7_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 7), 7, "short", p)

    def test_short_8_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 8), 8, "short", p)

    def test_short_9_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 9), 9, "short", p)

    def test_short_10_clips(self):
        p = self._apply_preset("short")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 10), 10, "short", p)

    def test_medium_1_clip(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 1), 1, "medium", p)

    def test_medium_2_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 2), 2, "medium", p)

    def test_medium_3_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 3), 3, "medium", p)

    def test_medium_4_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 4), 4, "medium", p)

    def test_medium_5_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 5), 5, "medium", p)

    def test_medium_6_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 6), 6, "medium", p)

    def test_medium_7_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 7), 7, "medium", p)

    def test_medium_8_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 8), 8, "medium", p)

    def test_medium_9_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 9), 9, "medium", p)

    def test_medium_10_clips(self):
        p = self._apply_preset("medium")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 10), 10, "medium", p)

    def test_long_1_clip(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 1), 1, "long", p)

    def test_long_2_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 2), 2, "long", p)

    def test_long_3_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 3), 3, "long", p)

    def test_long_4_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 4), 4, "long", p)

    def test_long_5_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 5), 5, "long", p)

    def test_long_6_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 6), 6, "long", p)

    def test_long_7_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 7), 7, "long", p)

    def test_long_8_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 8), 8, "long", p)

    def test_long_9_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 9), 9, "long", p)

    def test_long_10_clips(self):
        p = self._apply_preset("long")
        self._check_zones(pipeline.find_manual_clip_zones(self.duration, 10), 10, "long", p)


if __name__ == "__main__":
    unittest.main()
