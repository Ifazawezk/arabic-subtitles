import json

import pytest

from arsub import formats as F
from arsub import text as T
from arsub.builder import Cue, Word


@pytest.fixture
def cues():
    return [
        Cue(0.0, 2.5, ["مرحبا بكم"]),
        Cue(3.0, 6.25, ["في عام 2026 زاد العدد", "بنسبة 15%."]),
        Cue(3725.5, 3727.0, ["Python <3 & FastAPI"]),
    ]


def test_timestamps():
    assert F.format_timestamp(0) == "00:00:00,000"
    assert F.format_timestamp(3725.5) == "01:02:05,500"
    assert F.format_timestamp(59.9996) == "00:01:00,000"
    assert F.format_timestamp(1.25, ".") == "00:00:01.250"
    assert F.format_timestamp(-1) == "00:00:00,000"


def test_srt(cues):
    srt = F.to_srt(cues, rtl_marks="none")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nمرحبا بكم\n")
    assert "2\n00:00:03,000 --> 00:00:06,250\nفي عام 2026 زاد العدد\nبنسبة 15%.\n" in srt
    assert "3\n01:02:05,500 --> 01:02:07,000\n" in srt


def test_srt_rtl_marks_only_on_arabic_lines(cues):
    srt = F.to_srt(cues, rtl_marks="embed")
    assert T.RLE + "مرحبا بكم" + T.PDF in srt
    assert "\nPython <3 & FastAPI\n" in srt


def test_vtt(cues):
    vtt = F.to_vtt(cues, rtl_marks="none")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n")
    assert "Python &lt;3 &amp; FastAPI" in vtt


def test_round_trip_srt_and_vtt(cues):
    for content in (F.to_srt(cues), F.to_vtt(cues)):
        back = F.parse_srt(content)
        assert [c.lines for c in back] == [c.lines for c in cues]
        assert [round(c.start, 3) for c in back] == [c.start for c in cues]
        assert [round(c.end, 3) for c in back] == [c.end for c in cues]


def test_parse_srt_tolerates_bom_crlf_and_tags():
    content = "\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\n<i>مرحبا</i>\r\n\r\n2\r\n00:00:03,5 --> 00:00:04,000\r\nثانيا\r\n"
    back = F.parse_srt(content)
    assert [c.text for c in back] == ["مرحبا", "ثانيا"]
    assert back[1].start == 3.5


def test_parse_short_vtt_times():
    back = F.parse_srt("WEBVTT\n\n00:01.000 --> 00:02.500\nنعم\n")
    assert back[0].start == 1.0 and back[0].end == 2.5


def test_ass_is_rtl_safe(cues):
    ass = F.to_ass(cues, width=1280, height=720)
    assert "PlayResX: 1280" in ass and "PlayResY: 720" in ass
    style = next(line for line in ass.splitlines() if line.startswith("Style: Default"))
    assert style.endswith(",-1")  # Encoding -1 = detect paragraph direction
    assert "Dialogue: 0,0:00:03.00,0:00:06.25,Default,,0,0,0,," in ass
    assert T.RLE + "في عام 2026 زاد العدد" + T.PDF + "\\N" in ass


def test_ass_scales_with_height():
    small = F.to_ass([Cue(0, 1, ["نص"])], F.AssStyle(font_size=64), 1280, 720)
    style = next(line for line in small.splitlines() if line.startswith("Style: Default"))
    assert ",Noto Sans Arabic,43," in style


def test_ass_escapes_override_braces():
    ass = F.to_ass([Cue(0, 1, ["{\\b1}نص"])], rtl_marks="none")
    assert "\\{\\\\b1\\}نص" in ass


def test_ass_second_track_on_top():
    ass = F.to_ass([Cue(0, 1, ["مرحبا"])], second_track=[Cue(0, 1, ["Hello"])])
    assert ",Top,,0,0,0,,Hello" in ass


def test_txt_and_json(tmp_path, cues):
    assert F.to_txt(cues).splitlines()[1] == "في عام 2026 زاد العدد بنسبة 15%."
    words = [Word(0.0, 0.5, "مرحبا", 0.9)]
    data = json.loads(F.to_json(cues, words, {"model": "small"}))
    assert data["meta"]["model"] == "small"
    assert data["words"][0] == {"start": 0.0, "end": 0.5, "text": "مرحبا", "p": 0.9}
    path = tmp_path / "x.json"
    path.write_text(F.to_json(cues, words), encoding="utf-8")
    w2, c2, _ = F.load_json(path)
    assert w2[0].text == "مرحبا" and c2[1].lines == cues[1].lines


def test_write_all_formats(tmp_path, cues):
    for fmt in F.FORMATS:
        path = F.write(cues, fmt, tmp_path / f"out.{fmt}")
        assert path.stat().st_size > 0
    raw = (tmp_path / "out.srt").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM for old Windows players
    with pytest.raises(ValueError):
        F.write(cues, "docx", tmp_path / "out.docx")
