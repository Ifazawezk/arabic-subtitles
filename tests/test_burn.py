"""Burn-in: command construction with fakes, and one real ffmpeg run when available."""

import subprocess

import pytest

from arsub import burn as B
from arsub import formats as F
from arsub.builder import Cue


def test_find_ffmpeg_env_wins(monkeypatch, tmp_path):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_text("x")
    monkeypatch.setenv("ARSUB_FFMPEG", str(fake))
    assert B.find_ffmpeg() == str(fake)


def test_find_ffmpeg_path_then_package(monkeypatch):
    monkeypatch.delenv("ARSUB_FFMPEG", raising=False)
    monkeypatch.setattr(B.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert B.find_ffmpeg() == "/usr/bin/ffmpeg"


def test_command_uses_relative_names_only():
    cmd = B.build_command("ffmpeg", r"C:\Users\ناس\in put.mp4", r"C:\out\o.mp4")
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "ass=subs.ass:fontsdir=fonts"
    assert cmd[cmd.index("-i") + 1] == r"C:\Users\ناس\in put.mp4"
    assert cmd[-1] == r"C:\out\o.mp4"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "+faststart" in cmd


def test_command_for_audio_only_draws_a_background():
    cmd = B.build_command("ffmpeg", "talk.mp3", "o.mp4", audio_only=True, size=(1280, 720))
    assert "lavfi" in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "showwaves" in graph and "ass=subs.ass" in graph
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_command_nvenc():
    cmd = B.build_command("ffmpeg", "a.mp4", "b.mkv", video_codec="h264_nvenc", crf=23)
    assert cmd[cmd.index("-cq") + 1] == "29"  # crf + 6, see the comment in build_command
    assert cmd[cmd.index("-b:v") + 1] == "0"  # without this, -cq does not control the file size
    assert "-crf" not in cmd and "+faststart" not in cmd


def test_subtitle_filter_needs_libass(monkeypatch):
    monkeypatch.setattr(B, "ffmpeg_filters", lambda f: frozenset({"scale"}))
    with pytest.raises(B.FFmpegError, match="libass"):
        B.subtitle_filter("ffmpeg")


def test_subtitle_filter_prefers_ass(monkeypatch):
    monkeypatch.setattr(B, "ffmpeg_filters", lambda f: frozenset({"ass", "subtitles"}))
    assert B.subtitle_filter("ffmpeg") == "ass"


def test_is_audio_only_by_extension():
    assert B.is_audio_only("talk.mp3")
    assert B.is_audio_only("TALK.WAV")


def test_audio_only_trusts_the_contents_over_the_extension(monkeypatch):
    monkeypatch.setattr(B, "probe", lambda p: B.MediaInfo(has_video=True))
    assert not B.is_audio_only("video-in-disguise.ogg")


def test_fonts_ship_with_the_package():
    assert (B.FONTS_DIR / "NotoSansArabic-Bold.ttf").is_file()
    assert (B.FONTS_DIR / "OFL.txt").is_file()


def real_ffmpeg():
    ff = B.find_ffmpeg()
    if not ff:
        return None
    try:
        return ff if "ass" in B.ffmpeg_filters(ff) else None
    except OSError:
        return None


@pytest.mark.skipif(real_ffmpeg() is None, reason="no ffmpeg with libass here")
def test_real_burn_changes_pixels(tmp_path):
    """Burn into a 1 second clip and check the subtitle really reached the frame."""
    ff = real_ffmpeg()
    src = tmp_path / "ملف.mp4"  # a non-ASCII name on purpose
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=1:r=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    ass = F.to_ass([Cue(0.0, 1.0, ["مرحبا بكم؟"])], width=320, height=180)
    out = tmp_path / "out.mp4"
    seen = []
    B.burn(src, ass, out, ffmpeg=ff, preset="ultrafast", progress=lambda f, m: seen.append(f))
    assert out.stat().st_size > 0
    assert seen[-1] == 1.0

    import av

    with av.open(str(out)) as c:
        frame = next(c.decode(video=0)).to_ndarray(format="gray")
    assert frame.max() > 200  # black input, so bright pixels can only be the text
