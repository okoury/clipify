"""
Comprehensive tests for caption/export pipeline in clipify-pipeline.py
Run: python -m pytest tests/test_captions.py -v
"""
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch, call


# ── Load the pipeline module (hyphenated filename) ─────────────────────────────
def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "clipify_pipeline",
        os.path.join(os.path.dirname(__file__), "..", "clipify-pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    # Stub out heavy ML imports before exec
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock()
    sys.modules.setdefault("faster_whisper", faster_whisper_stub)

    vader_stub = types.ModuleType("vaderSentiment")
    vader_sa   = types.ModuleType("vaderSentiment.vaderSentiment")
    vader_sa.SentimentIntensityAnalyzer = MagicMock(return_value=MagicMock(
        polarity_scores=MagicMock(return_value={"compound": 0.0, "neg": 0.0, "pos": 0.0, "neu": 1.0})
    ))
    sys.modules.setdefault("vaderSentiment", vader_stub)
    sys.modules.setdefault("vaderSentiment.vaderSentiment", vader_sa)

    spec.loader.exec_module(mod)
    return mod


pipeline = _load_pipeline()


# ── Helpers: fake Whisper word/segment objects ─────────────────────────────────
def _word(word, start, end):
    w = MagicMock()
    w.word  = word
    w.start = start
    w.end   = end
    return w


def _seg(words):
    s = MagicMock()
    s.words = words
    s.text  = " ".join(w.word for w in words)
    s.start = words[0].start if words else 0
    s.end   = words[-1].end  if words else 0
    return s


# ══════════════════════════════════════════════════════════════════════════════
class TestTsAss(unittest.TestCase):
    """_ts_ass converts seconds to h:mm:ss.cc ASS format."""

    def test_zero(self):
        self.assertEqual(pipeline._ts_ass(0.0), "0:00:00.00")

    def test_one_second(self):
        self.assertEqual(pipeline._ts_ass(1.0), "0:00:01.00")

    def test_centiseconds(self):
        self.assertEqual(pipeline._ts_ass(1.235), "0:00:01.24")  # rounds to nearest cs

    def test_one_minute(self):
        self.assertEqual(pipeline._ts_ass(60.0), "0:01:00.00")

    def test_one_hour(self):
        self.assertEqual(pipeline._ts_ass(3600.0), "1:00:00.00")

    def test_complex(self):
        self.assertEqual(pipeline._ts_ass(3723.45), "1:02:03.45")

    def test_negative_clamps_to_zero(self):
        self.assertEqual(pipeline._ts_ass(-5.0), "0:00:00.00")


# ══════════════════════════════════════════════════════════════════════════════
class TestChunkWords(unittest.TestCase):
    """_chunk_words splits on gap > 1.5s or max size words."""

    def _w(self, start, end):
        return {"word": "x", "start": start, "end": end}

    def test_simple_size_split(self):
        words = [self._w(i, i + 0.5) for i in range(10)]
        chunks = pipeline._chunk_words(words, size=5, gap_thresh=1.5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(len(chunks[1]), 5)

    def test_gap_triggers_new_chunk(self):
        words = [
            self._w(0.0, 0.4), self._w(0.5, 0.9),
            self._w(3.0, 3.4), self._w(3.5, 3.9),  # gap of 2.1s before word at 3.0
        ]
        chunks = pipeline._chunk_words(words, size=5, gap_thresh=1.5)
        self.assertEqual(len(chunks), 2)

    def test_no_gap_no_split(self):
        words = [self._w(i * 0.5, i * 0.5 + 0.4) for i in range(4)]
        chunks = pipeline._chunk_words(words, size=5, gap_thresh=1.5)
        self.assertEqual(len(chunks), 1)

    def test_empty_input(self):
        self.assertEqual(pipeline._chunk_words([]), [])

    def test_single_word(self):
        chunks = pipeline._chunk_words([self._w(0, 1)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 1)


# ══════════════════════════════════════════════════════════════════════════════
class TestCollectWordsForZone(unittest.TestCase):
    """collect_words_for_zone produces relative timestamps with no negatives."""

    def test_basic_relative_offset(self):
        words = [_word("hello", 5.0, 5.5), _word("world", 5.6, 6.0)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["start"], 0.0)
        self.assertAlmostEqual(result[0]["end"],   0.5)
        self.assertAlmostEqual(result[1]["start"], 0.6)

    def test_no_negative_starts(self):
        # Word starts before zone — should clamp start to 0
        words = [_word("early", 4.8, 5.2)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 0.0)
        self.assertGreater(result[0]["end"], 0.0)

    def test_word_entirely_before_zone_excluded(self):
        words = [_word("before", 1.0, 2.0)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
        self.assertEqual(result, [])

    def test_word_entirely_after_zone_excluded(self):
        words = [_word("after", 11.0, 12.0)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
        self.assertEqual(result, [])

    def test_word_at_exact_zone_boundary_excluded(self):
        # Word starts exactly at zone_end — should be excluded (strict <)
        words = [_word("boundary", 10.0, 10.5)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
        self.assertEqual(result, [])

    def test_word_end_clamped_to_zone_end(self):
        # Word extends past zone_end — end should be clamped
        words = [_word("overflow", 9.5, 10.8)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=5.0, zone_end=10.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["end"], 5.0)  # 10.0 - 5.0

    def test_ordering_preserved(self):
        words = [_word(f"w{i}", i * 1.0, i * 1.0 + 0.5) for i in range(5)]
        seg   = _seg(words)
        result = pipeline.collect_words_for_zone([seg], zone_start=0.0, zone_end=10.0)
        starts = [r["start"] for r in result]
        self.assertEqual(starts, sorted(starts))


# ══════════════════════════════════════════════════════════════════════════════
class TestWriteAssSubtitles(unittest.TestCase):
    """_write_ass_subtitles writes valid ASS files with karaoke tags."""

    def _make_words(self):
        return [
            {"word": "Hello", "start": 0.0,  "end": 0.5},
            {"word": "world", "start": 0.6,  "end": 1.1},
            {"word": "this",  "start": 1.2,  "end": 1.5},
            {"word": "is",    "start": 1.6,  "end": 1.8},
            {"word": "great", "start": 1.9,  "end": 2.3},
        ]

    def test_file_written(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            path = f.name
        try:
            pipeline._write_ass_subtitles(self._make_words(), path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_ass_sections_present(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
            path = f.name
        try:
            pipeline._write_ass_subtitles(self._make_words(), path)
            content = open(path, encoding="utf-8").read()
            self.assertIn("[Script Info]", content)
            self.assertIn("[V4+ Styles]", content)
            self.assertIn("[Events]", content)
            self.assertIn("Dialogue:", content)
        finally:
            os.unlink(path)

    def test_karaoke_tags_present(self):
        words = [{"word": "test", "start": 0.0, "end": 0.5}]
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
            path = f.name
        try:
            pipeline._write_ass_subtitles(words, path)
            content = open(path, encoding="utf-8").read()
            self.assertIn("{\\k", content)
        finally:
            os.unlink(path)

    def test_empty_words_produces_valid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
            path = f.name
        try:
            pipeline._write_ass_subtitles([], path)
            content = open(path, encoding="utf-8").read()
            self.assertIn("[Script Info]", content)
        finally:
            os.unlink(path)

    def test_gold_colour_in_style(self):
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
            path = f.name
        try:
            pipeline._write_ass_subtitles(self._make_words(), path)
            content = open(path, encoding="utf-8").read()
            # Gold in BGR hex as used in ASS: &H0000D7FF
            self.assertIn("0000D7FF", content)
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
class TestBurnCaptionsWatermark(unittest.TestCase):
    """_burn_captions_watermark calls FFmpeg with correct filter arguments."""

    def test_ffmpeg_called_with_ass_filter(self):
        words = [{"word": "hi", "start": 0.0, "end": 0.5}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(pipeline, "_write_ass_subtitles"):
            result = pipeline._burn_captions_watermark(
                "input.mp4", 0.0, 10.0, words, "output.mp4"
            )
        self.assertTrue(result)
        self.assertTrue(mock_ffmpeg.called)
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf = cmd[vf_idx + 1]
        self.assertIn("ass=", vf)
        self.assertIn("drawtext", vf)

    def test_watermark_text_is_snipflow(self):
        words = [{"word": "hi", "start": 0.0, "end": 0.5}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(pipeline, "_write_ass_subtitles"):
            pipeline._burn_captions_watermark(
                "input.mp4", 0.0, 10.0, words, "output.mp4"
            )
        cmd = mock_ffmpeg.call_args[0][0]
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("Snipflow", vf)

    def test_watermark_bottom_right_position(self):
        words = [{"word": "hi", "start": 0.0, "end": 0.5}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(pipeline, "_write_ass_subtitles"):
            pipeline._burn_captions_watermark(
                "input.mp4", 0.0, 10.0, words, "output.mp4"
            )
        cmd = mock_ffmpeg.call_args[0][0]
        vf = cmd[cmd.index("-vf") + 1]
        # Bottom-right: x=w-tw-*, y=h-th-*
        self.assertIn("x=w-tw", vf)
        self.assertIn("y=h-th", vf)

    def test_returns_false_on_ffmpeg_failure(self):
        words = [{"word": "hi", "start": 0.0, "end": 0.5}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=False), \
             patch.object(pipeline, "_write_ass_subtitles"):
            result = pipeline._burn_captions_watermark(
                "input.mp4", 0.0, 10.0, words, "output.mp4"
            )
        self.assertFalse(result)

    def test_start_offset_passed_to_ffmpeg(self):
        words = [{"word": "hi", "start": 0.0, "end": 0.5}]
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(pipeline, "_write_ass_subtitles"):
            pipeline._burn_captions_watermark(
                "input.mp4", 30.5, 15.0, words, "output.mp4"
            )
        cmd = mock_ffmpeg.call_args[0][0]
        ss_idx = cmd.index("-ss")
        self.assertEqual(cmd[ss_idx + 1], "30.5")


# ══════════════════════════════════════════════════════════════════════════════
class TestStreamCopyClip(unittest.TestCase):
    """_stream_copy_clip uses stream copy with re-encode fallback."""

    def test_returns_true_on_success(self):
        with patch.object(pipeline, "_run_ffmpeg", return_value=True) as mock_ffmpeg:
            result = pipeline._stream_copy_clip("in.mp4", 10.0, 30.0, "out.mp4")
        self.assertTrue(result)
        cmd = mock_ffmpeg.call_args[0][0]
        self.assertIn("-c", cmd)
        self.assertIn("copy", cmd)

    def test_fallback_on_stream_copy_failure(self):
        calls = []
        def side_effect(cmd, **kwargs):
            calls.append(cmd)
            return len(calls) > 1  # first call fails, second succeeds
        with patch.object(pipeline, "_run_ffmpeg", side_effect=side_effect):
            result = pipeline._stream_copy_clip("in.mp4", 10.0, 30.0, "out.mp4")
        self.assertTrue(result)
        self.assertEqual(len(calls), 2)
        # Second call should use libx264
        self.assertIn("libx264", calls[1])

    def test_returns_false_when_both_fail(self):
        with patch.object(pipeline, "_run_ffmpeg", return_value=False):
            result = pipeline._stream_copy_clip("in.mp4", 10.0, 30.0, "out.mp4")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
