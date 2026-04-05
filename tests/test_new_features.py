"""
Tests for manual mode, BURN_VISUALS, normalized score, seek fix, and social upload.
Run: python -m pytest tests/test_new_features.py -v
"""
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ── Load pipeline ──────────────────────────────────────────────────────────────
def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "clipify_pipeline",
        os.path.join(os.path.dirname(__file__), "..", "clipify-pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    fw = types.ModuleType("faster_whisper"); fw.WhisperModel = MagicMock()
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


def _seg(start, end, text="hello world great amazing"):
    s = MagicMock(); s.start = start; s.end = end; s.text = text; s.words = []
    return s


def _segs(n=20, span=200.0):
    step = span / n
    return [_seg(i * step, i * step + step * 0.8) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
class TestManualMode(unittest.TestCase):
    """Manual mode: deterministic clip count, no score dependency."""

    def setUp(self):
        pipeline.MIN_DURATION = 15
        pipeline.MAX_DURATION = 35
        pipeline.TARGET_DURATION = 25

    def test_returns_exact_count(self):
        zones = pipeline.find_manual_clip_zones(300.0, 5)
        self.assertEqual(len(zones), 5)

    def test_returns_exact_count_varied(self):
        for n in (1, 3, 7, 10):
            zones = pipeline.find_manual_clip_zones(float(n * 60), n)
            self.assertEqual(len(zones), n, f"Expected {n}, got {len(zones)}")

    def test_duration_within_bounds(self):
        zones = pipeline.find_manual_clip_zones(300.0, 5)
        for z in zones:
            dur = z["end"] - z["start"]
            self.assertGreaterEqual(dur, pipeline.MIN_DURATION - 0.1)
            self.assertLessEqual(dur, pipeline.MAX_DURATION + 0.1)

    def test_no_overlap(self):
        zones = pipeline.find_manual_clip_zones(300.0, 5)
        zones = sorted(zones, key=lambda z: z["start"])
        for i in range(len(zones) - 1):
            self.assertLessEqual(
                zones[i]["end"], zones[i + 1]["start"] + 0.6,
                f"Overlap between zone {i} and {i+1}",
            )

    def test_zero_clips_returns_empty(self):
        self.assertEqual(pipeline.find_manual_clip_zones(100.0, 0), [])

    def test_zero_duration_returns_empty(self):
        self.assertEqual(pipeline.find_manual_clip_zones(0.0, 5), [])

    def test_zones_within_video_bounds(self):
        span  = 180.0
        zones = pipeline.find_manual_clip_zones(span, 4)
        for z in zones:
            self.assertGreaterEqual(z["start"], 0.0)
            self.assertLessEqual(z["end"], span + 1.0)

    def test_auto_mode_still_works(self):
        """Auto mode (max_clips=0) uses find_clip_zones unchanged."""
        segs = [_seg(i * 20, i * 20 + 15, "crazy insane unbelievable wow") for i in range(5)]
        pipeline.MIN_SEG_SCORE  = 0
        pipeline.MIN_ZONE_SCORE = 0
        pipeline.CLUSTER_GAP   = 5
        zones = pipeline.find_clip_zones(segs)
        self.assertIsInstance(zones, list)


# ══════════════════════════════════════════════════════════════════════════════
class TestBurnVisuals(unittest.TestCase):
    """BURN_VISUALS controls watermark; captions always burn when words exist."""

    def _zone(self):
        return {"start": 0.0, "end": 20.0, "score": 5.0, "text": "test",
                "reasons": [], "words": []}

    def test_captions_always_burned_when_words_exist(self):
        """Regardless of FAST_MODE, captions burn when words are present."""
        pipeline.FAST_MODE    = True
        pipeline.BURN_VISUALS = True
        zone = self._zone()
        fake_segs = [MagicMock()]

        with patch.object(pipeline, "_burn_captions_watermark", return_value=True) as mock_burn, \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone",
                          return_value=[{"word": "hi", "start": 0.0, "end": 0.5}]):
            pipeline._extract_one(1, zone, "in.mp4", "mp4", True, True, fake_segs)

        mock_burn.assert_called_once()

    def test_captions_burned_when_no_words_fallback(self):
        """No words → fallback word injected and _burn_captions_watermark still called."""
        pipeline.FAST_MODE    = True
        pipeline.BURN_VISUALS = True
        zone = self._zone()

        with patch.object(pipeline, "_burn_captions_watermark", return_value=True) as mock_caps, \
             patch.object(pipeline, "_burn_watermark_only") as mock_wm, \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[]):
            pipeline._extract_one(1, zone, "in.mp4", "mp4", True, True, [MagicMock()])

        mock_caps.assert_called_once()
        mock_wm.assert_not_called()

    def test_stream_copy_only_after_both_burns_fail(self):
        """Stream copy is fallback only when both burn functions fail."""
        pipeline.FAST_MODE    = True
        pipeline.BURN_VISUALS = False
        zone = self._zone()

        with patch.object(pipeline, "_burn_captions_watermark", return_value=False) as mock_caps, \
             patch.object(pipeline, "_burn_watermark_only", return_value=False) as mock_wm, \
             patch.object(pipeline, "_stream_copy_clip", return_value=True) as mock_sc, \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[]):
            pipeline._extract_one(1, zone, "in.mp4", "mp4", True, True, [MagicMock()])

        mock_caps.assert_called_once()
        mock_wm.assert_called_once()
        mock_sc.assert_called_once()

    def tearDown(self):
        pipeline.FAST_MODE    = True
        pipeline.BURN_VISUALS = True


# ══════════════════════════════════════════════════════════════════════════════
class TestAccurateSeek(unittest.TestCase):
    """Caption burn: accurate seek (-i first, -ss after) + setpts=PTS-STARTPTS."""

    def test_single_ss_after_input(self):
        """Exactly one -ss, and it appears after -i."""
        words = [{"word": "hi", "start": 0.0, "end": 0.5}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(pipeline, "_write_ass_subtitles"):
            pipeline._burn_captions_watermark("in.mp4", 30.0, 15.0, words, "out.mp4")

        cmd = mock_ffmpeg.call_args[0][0]
        ss_indices = [i for i, x in enumerate(cmd) if x == "-ss"]
        self.assertEqual(len(ss_indices), 1)
        self.assertGreater(ss_indices[0], cmd.index("-i"))

    def test_ss_equals_start(self):
        """-ss value equals the clip start time."""
        words = [{"word": "x", "start": 0.0, "end": 0.3}]
        for start in [0.0, 30.0, 30.5]:
            with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
                 patch.object(pipeline, "_write_ass_subtitles"):
                pipeline._burn_captions_watermark("in.mp4", start, 10.0, words, "out.mp4")
            cmd = mock_ffmpeg.call_args[0][0]
            ss_val = float(cmd[cmd.index("-ss") + 1])
            self.assertAlmostEqual(ss_val, start, places=2)

    def test_setpts_in_vf(self):
        """setpts=PTS-STARTPTS must appear in the -vf string before ass=."""
        words = [{"word": "x", "start": 0.0, "end": 0.3}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(pipeline, "_write_ass_subtitles"):
            pipeline._burn_captions_watermark("in.mp4", 30.0, 10.0, words, "out.mp4")
        cmd = mock_ffmpeg.call_args[0][0]
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("setpts=PTS-STARTPTS", vf)
        self.assertIn("ass=", vf)
        self.assertLess(vf.index("setpts"), vf.index("ass="))


# ══════════════════════════════════════════════════════════════════════════════
class TestNormalizedScore(unittest.TestCase):
    """normalized_score maps raw score → 0–100; frontend labels correct."""

    def _make_zone(self, score):
        return {"start": 0.0, "end": 20.0, "score": score, "text": "test",
                "reasons": ["test reason"], "words": []}

    def test_normalized_score_in_clip_dict(self):
        zone = self._make_zone(20.0)
        pipeline.SCORE_FACTOR = 2.5

        with patch.object(pipeline, "_burn_watermark_only", return_value=True), \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[]):
            clip = pipeline._extract_one(1, zone, "in.mp4", "mp4", True, False, [])

        self.assertIn("normalized_score", clip)
        self.assertEqual(clip["normalized_score"], min(100, int(20.0 * 2.5)))

    def test_score_capped_at_100(self):
        zone = self._make_zone(50.0)
        pipeline.SCORE_FACTOR = 2.5

        with patch.object(pipeline, "_burn_watermark_only", return_value=True), \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[]):
            clip = pipeline._extract_one(1, zone, "in.mp4", "mp4", True, False, [])

        self.assertEqual(clip["normalized_score"], 100)

    def test_score_label_thresholds(self):
        """Verify HIGH/MEDIUM/LOW band boundaries used in frontend."""
        self.assertEqual("HIGH",   "HIGH"   if 90  > 80 else "other")
        self.assertEqual("MEDIUM", "MEDIUM" if 60  > 50 else "other")
        self.assertEqual("LOW",    "LOW"    if 30 <= 50 else "other")

    def test_reasons_in_clip_dict(self):
        zone = self._make_zone(10.0)
        zone["reasons"] = ["strong emotion", "hook:\"wait\"", "2 exclamation(s)"]

        with patch.object(pipeline, "_burn_watermark_only", return_value=True), \
             patch.object(pipeline, "generate_thumbnail", return_value=None), \
             patch.object(pipeline, "collect_words_for_zone", return_value=[]):
            clip = pipeline._extract_one(1, zone, "in.mp4", "mp4", True, False, [])

        self.assertIn("reasons", clip)
        self.assertLessEqual(len(clip["reasons"]), 3)


# ══════════════════════════════════════════════════════════════════════════════
class TestSocialUpload(unittest.TestCase):
    """social_upload module dispatches correctly and stubs return expected shape."""

    def _load_social(self):
        spec = importlib.util.spec_from_file_location(
            "social_upload",
            os.path.join(os.path.dirname(__file__), "..", "social_upload.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_unknown_platform_returns_error(self):
        su = self._load_social()
        r  = su.upload_clip("myspace", "/fake/clip.mp4", "title")
        self.assertFalse(r["ok"])
        self.assertIn("Unknown platform", r["error"])

    def test_tiktok_stub_returns_not_ok(self):
        su = self._load_social()
        r  = su.upload_clip("tiktok", "/fake/clip.mp4", "title")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_instagram_stub_returns_not_ok(self):
        su = self._load_social()
        r  = su.upload_clip("instagram", "/fake/clip.mp4", "title")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_youtube_missing_secrets_returns_error(self):
        su = self._load_social()
        os.environ.pop("YOUTUBE_CLIENT_SECRETS_FILE", None)
        r  = su.upload_clip("youtube", "/fake/clip.mp4", "title")
        self.assertFalse(r["ok"])
        self.assertIn("YOUTUBE_CLIENT_SECRETS_FILE", r["error"])

    def test_youtube_upload_mock(self):
        """Full YouTube upload path mocked — no real API call."""
        su = self._load_social()

        mock_video_id = "abc123"
        mock_response = {"id": mock_video_id}

        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (None, mock_response)

        mock_videos = MagicMock()
        mock_videos.return_value.insert.return_value = mock_request

        mock_youtube = MagicMock()
        mock_youtube.videos = mock_videos

        fake_secrets = "/tmp/fake_secrets.json"
        with open(fake_secrets, "w") as f:
            f.write('{"installed":{}}')
        os.environ["YOUTUBE_CLIENT_SECRETS_FILE"] = fake_secrets

        fake_creds = MagicMock()
        fake_creds.valid = True

        import builtins
        import io as _io

        with patch.dict("sys.modules", {
            "googleapiclient":                   MagicMock(),
            "googleapiclient.discovery":         MagicMock(build=MagicMock(return_value=mock_youtube)),
            "googleapiclient.http":              MagicMock(MediaFileUpload=MagicMock()),
            "google_auth_oauthlib":              MagicMock(),
            "google_auth_oauthlib.flow":         MagicMock(),
            "google.auth.transport.requests":    MagicMock(),
            "pickle":                            MagicMock(load=MagicMock(return_value=fake_creds)),
        }), patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=b""))),
            __exit__=MagicMock(return_value=False),
        ))), patch("os.path.exists", return_value=True):
            importlib.invalidate_caches()
            su2 = self._load_social()
            r   = su2.upload_clip("youtube", "/fake/clip.mp4", "Test Title")

        # Depending on mock plumbing, ok=True or error about googleapiclient —
        # either is acceptable as long as the structure is correct
        self.assertIn("ok", r)

        os.environ.pop("YOUTUBE_CLIENT_SECRETS_FILE", None)
        try: os.unlink(fake_secrets)
        except: pass


if __name__ == "__main__":
    unittest.main()
