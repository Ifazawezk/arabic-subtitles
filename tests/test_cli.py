"""CLI and pipeline end to end, with the transcriber and ffmpeg replaced by fakes."""

import json

import pytest

from arsub import cli, pipeline
from arsub.builder import Word
from arsub.transcribe import Transcript


def fake_transcript(task="transcribe", model="small"):
    if task == "translate" and "turbo" in model:
        words = [Word(0.0, 0.5, " مرحبا"), Word(0.6, 1.0, " بكم")]  # what the real turbo model does
    elif task == "translate":
        words = [Word(0.0, 0.5, " Hello"), Word(0.6, 1.0, " everyone."), Word(1.1, 1.5, " Welcome")]
    else:
        words = [Word(0.0, 0.5, " مرحبا"), Word(0.6, 1.0, " بكم ,"), Word(1.1, 1.5, " في"), Word(1.6, 2.0, " عام"), Word(2.1, 2.5, " 2026")]
    return Transcript(words=words, duration=3.0, device="cpu", model=model, compute_type="int8", elapsed=1.0)


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "talk.mp4"
    path.write_bytes(b"not really a video")
    return path


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake(audio, opts=None, progress=None):
        seen.append(opts)
        if progress:
            progress(1.0, "done")
        return fake_transcript(opts.task, "small" if opts.model == "auto" else opts.model)

    monkeypatch.setattr(pipeline, "transcribe", fake)
    monkeypatch.setattr(pipeline, "video_size", lambda p: (1280, 720))
    return seen


def test_default_writes_srt_next_to_input(media, calls, capsys):
    assert cli.main([str(media)]) == 0
    out = media.with_name("talk.ar.srt")
    text = out.read_text(encoding="utf-8-sig")
    assert "مرحبا بكم،" in text
    assert str(out) in capsys.readouterr().out


def test_several_formats_and_output_dir(media, calls, tmp_path):
    outdir = tmp_path / "subs"
    assert cli.main([str(media), "-f", "srt,vtt,ass,txt,json", "-o", str(outdir), "-q"]) == 0
    assert sorted(p.name for p in outdir.iterdir()) == [
        "talk.ar.ass", "talk.ar.json", "talk.ar.srt", "talk.ar.txt", "talk.ar.vtt",
    ]
    assert "PlayResX: 1280" in (outdir / "talk.ar.ass").read_text(encoding="utf-8")


def test_all_format_keyword(media, calls):
    assert cli.main([str(media), "-f", "all", "-q"]) == 0
    assert media.with_name("talk.ar.vtt").exists()


def test_options_reach_transcriber_and_builder(media, calls):
    args = [str(media), "-f", "txt", "-m", "tiny", "--device", "cpu", "--dialect", "gulf", "--digits", "arabic-indic",
            "--no-vad", "--beam-size", "1", "--batch-size", "4", "-q"]
    assert cli.main(args) == 0
    opts = calls[0]
    assert (opts.model, opts.device, opts.dialect, opts.vad, opts.beam_size, opts.batch_size) == (
        "tiny", "cpu", "gulf", False, 1, 4,
    )
    assert "٢٠٢٦" in media.with_name("talk.ar.txt").read_text(encoding="utf-8")


def test_rebuild_from_json_skips_the_model(media, calls, monkeypatch):
    assert cli.main([str(media), "-f", "json", "-q"]) == 0
    assert len(calls) == 1
    dump = media.with_name("talk.ar.json")
    assert cli.main([str(dump), "-f", "srt", "--max-chars", "12", "-q"]) == 0
    assert len(calls) == 1  # no second transcription
    assert media.with_name("talk.ar.srt").exists()


def test_translate_writes_english_track(media, calls):
    assert cli.main([str(media), "--translate", "-f", "srt", "-q"]) == 0
    assert [c.task for c in calls] == ["transcribe", "translate"]
    english = media.with_name("talk.en.srt").read_text(encoding="utf-8-sig")
    assert "Hello everyone." in english
    assert "\u202b" not in english


def test_translation_avoids_turbo(media, calls):
    assert cli.main([str(media), "--translate", "-m", "large-v3-turbo", "-q"]) == 0
    assert [c.model for c in calls] == ["large-v3-turbo", "medium"]
    assert "Hello everyone." in media.with_name("talk.en.srt").read_text(encoding="utf-8-sig")


def test_translation_keeps_a_good_model(media, calls):
    assert cli.main([str(media), "--translate", "-m", "large-v3", "-q"]) == 0
    assert [c.model for c in calls] == ["large-v3", "large-v3"]


def test_warns_when_translation_stays_arabic(media, calls, caplog):
    args = [str(media), "--translate", "-m", "large-v3-turbo", "--translate-model", "turbo-custom"]
    assert cli.main(args) == 0
    assert "returned Arabic" in caplog.text


def test_translate_drops_the_dialect_prompt(media, calls):
    assert cli.main([str(media), "--translate", "--dialect", "egyptian", "-q"]) == 0
    assert calls[0].dialect == "egyptian"
    assert calls[1].dialect is None


def burn_recorder(monkeypatch):
    burned = {}

    def fake_burn(media, ass_text, out, **kw):
        burned.update(media=media, ass=ass_text, out=out, **kw)
        return out

    monkeypatch.setattr(pipeline, "burn_video", fake_burn)
    return burned


def test_burn_passes_rtl_ass(media, calls, monkeypatch):
    burned = burn_recorder(monkeypatch)
    assert cli.main([str(media), "--burn", "--video-codec", "h264_nvenc", "--font-size", "50", "-q"]) == 0
    assert burned["out"].name == "talk.subbed.mp4"
    assert burned["video_codec"] == "h264_nvenc"
    assert ",-1\n" in burned["ass"]  # Encoding -1
    assert "\u202b" in burned["ass"]


def test_burn_both_tracks(media, calls, monkeypatch):
    burned = burn_recorder(monkeypatch)
    assert cli.main([str(media), "--translate", "--burn", "--burn-track", "both", "-q"]) == 0
    assert ",Top,,0,0,0,,Hello" in burned["ass"]


def test_burn_existing_subtitles_skips_the_model(media, calls, monkeypatch, tmp_path):
    burned = burn_recorder(monkeypatch)
    subs = tmp_path / "edited.srt"
    subs.write_text("1\n00:00:01,000 --> 00:00:02,000\nنص معدل\n", encoding="utf-8")
    assert cli.main([str(media), "--subs", str(subs), "--burn", "-q"]) == 0
    assert calls == []
    assert "نص معدل" in burned["ass"]


def test_subs_rejects_formats_it_cannot_read(media, calls, tmp_path, capsys):
    subs = tmp_path / "styled.ass"
    subs.write_text("[Script Info]", encoding="utf-8")
    assert cli.main([str(media), "--subs", str(subs), "--burn", "-q"]) == 1
    assert ".srt, .vtt or .json" in capsys.readouterr().err


def test_no_speech_warns(media, monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: Transcript(words=[], duration=3.0))
    monkeypatch.setattr(pipeline, "video_size", lambda p: (1280, 720))
    assert cli.main([str(media)]) == 0
    assert "no speech found" in caplog.text


def test_json_output_records_source(media, calls):
    assert cli.main([str(media), "-f", "json", "-q"]) == 0
    data = json.loads(media.with_name("talk.ar.json").read_text(encoding="utf-8"))
    assert data["meta"]["model"] == "small"
    assert data["meta"]["source"].endswith("talk.mp4")
    assert data["words"][0]["text"].strip() == "مرحبا"


def test_missing_file(capsys, tmp_path):
    assert cli.main([str(tmp_path / "nope.mp4")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_no_input(capsys):
    assert cli.main([]) == 2


def test_bad_format(media, calls):
    with pytest.raises(SystemExit):
        cli.main([str(media), "-f", "docx"])


def test_burn_track_needs_translate(media, calls, capsys):
    assert cli.main([str(media), "--burn", "--burn-track", "en"]) == 2
    assert "--translate" in capsys.readouterr().err


def test_translate_cannot_use_subs(media, calls, tmp_path, capsys):
    subs = tmp_path / "x.srt"
    subs.write_text("", encoding="utf-8")
    assert cli.main([str(media), "--translate", "--subs", str(subs)]) == 2


def test_failure_is_reported_not_raised(media, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    assert cli.main([str(media), "-q"]) == 1
    assert "model exploded" in capsys.readouterr().err


def test_parse_formats():
    assert cli.parse_formats("SRT, .vtt,srt") == ("srt", "vtt")
    assert cli.parse_formats("") == ("srt",)
