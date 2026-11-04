"""Editor helpers. These do not need Gradio to be installed."""

import pytest

from arsub import web
from arsub.builder import BuildOptions, Cue


def test_parse_time_forms():
    assert web.parse_time("83.5") == 83.5
    assert web.parse_time("01:23.5") == 83.5
    assert web.parse_time("00:01:23,500") == 83.5
    assert web.parse_time(12) == 12.0
    with pytest.raises(ValueError):
        web.parse_time("")
    with pytest.raises(ValueError):
        web.parse_time("soon")


def test_rows_round_trip_keeps_line_breaks():
    cues = [Cue(0.0, 2.0, ["السطر الأول", "السطر الثاني"]), Cue(3.0, 5.0, ["سطر واحد"])]
    back = web.rows_to_cues(web.cues_to_rows(cues))
    assert [c.lines for c in back] == [c.lines for c in cues]
    assert [c.start for c in back] == [0.0, 3.0]


def test_editing_text_resplits_and_fixes_punctuation():
    rows = [[1, "00:00:00.000", "00:00:04.000", "هذا نص طويل جدا كتبه المستخدم في الجدول , ويجب تقسيمه على سطرين", 0]]
    cue = web.rows_to_cues(rows, BuildOptions(max_chars=42))[0]
    assert len(cue.lines) == 2
    assert "،" in cue.text


def test_empty_rows_are_removed_and_order_is_by_time():
    rows = [
        [1, "5", "6", "ثانيا", 0],
        [2, "1", "2", "   ", 0],
        [3, "0", "1", "أولا", 0],
        [4, "7", "8", None, 0],
    ]
    assert [c.text for c in web.rows_to_cues(rows)] == ["أولا", "ثانيا"]


def test_dataframe_input_is_accepted():
    class Frame:
        columns = web.COLUMNS

        class values:  # noqa: N801
            @staticmethod
            def tolist():
                return [[1, "0", "1", "نص", 0]]

    assert web.rows_to_cues(Frame())[0].text == "نص"


def test_summary_mentions_speed():
    opts = BuildOptions()
    meta = {"device": "cuda", "model": "small", "duration": 60.0, "elapsed": 6.0}
    text = web.summary([Cue(0, 2, ["مرحبا بكم"])], opts, meta)
    assert "1 cues" in text and "small on cuda" in text and "10.0x realtime" in text


def test_export_files(tmp_path):
    files = web.export_files([Cue(0, 1, ["نص"])], "clip", tmp_path / "out", {"width": 640, "height": 360})
    assert [p.rsplit(".", 1)[-1] for p in files] == ["srt", "vtt", "ass"]
