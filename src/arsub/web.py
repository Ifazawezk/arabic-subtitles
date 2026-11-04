"""Local web UI: drop a file, watch the progress bar, fix the text, export.

Runs on 127.0.0.1 only. Gradio's analytics are switched off and the theme uses
system fonts, so the page makes no requests to outside servers.

    arsub-web
"""

from __future__ import annotations

import argparse
import os
import tempfile
from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import quote

from . import formats as F
from .builder import BuildOptions, Cue, cue_from_text, stats
from .burn import burn as burn_video
from .burn import is_audio_only
from .dialects import DIALECTS
from .pipeline import run, video_size
from .text import DIGIT_STYLES, display_len
from .transcribe import MODELS, TranscribeOptions, cuda_device_count

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

COLUMNS = ["#", "Start", "End", "Text", "CPS"]

CSS = """
#arsub-table [role="gridcell"][data-col="3"], #arsub-table [role="gridcell"][data-col="3"] * {
  direction: rtl; text-align: right;
  font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, sans-serif; font-size: 1.05rem; }
footer { display: none !important; }
"""

INTRO = """
# arsub
Arabic subtitles, on this computer. Drop a video or audio file, press **Make subtitles**,
then fix anything you like in the table and export.
"""


def parse_time(value: object) -> float:
    """``83.5``, ``01:23.5`` or ``00:01:23,500`` to seconds."""
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("empty time")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def cues_to_rows(cues: list[Cue]) -> list[list]:
    """The table the editor shows. Line breaks are shown as `` / ``."""
    rows = []
    for i, cue in enumerate(cues, 1):
        text = " / ".join(cue.lines)
        rows.append(
            [i, F.format_timestamp(cue.start, "."), F.format_timestamp(cue.end, "."), text, round(cue.cps, 1)]
        )
    return rows


def rows_to_cues(rows, opts: BuildOptions | None = None) -> list[Cue]:
    """Read the edited table back.

    A `` / `` in the text forces a line break. Without one the text is split
    again with the normal Arabic rules. Empty rows are removed.
    """
    opts = opts or BuildOptions()
    if hasattr(rows, "values") and hasattr(rows, "columns"):  # pandas DataFrame
        rows = rows.values.tolist()
    cues: list[Cue] = []
    for row in rows or []:
        if len(row) < 4:
            continue
        text = str(row[3] if row[3] is not None else "").strip()
        if not text or text.lower() == "nan":
            continue
        start, end = parse_time(row[1]), parse_time(row[2])
        if " / " in text:
            cue = cue_from_text(start, end, text.replace(" / ", " "), opts)
            cue.lines = [cue_from_text(start, end, part, opts).text.replace("\n", " ") for part in text.split(" / ")]
        else:
            cue = cue_from_text(start, end, text, opts)
        cues.append(cue)
    return sorted(cues, key=lambda c: c.start)


def summary(cues: list[Cue], opts: BuildOptions, meta: dict | None = None) -> str:
    st = stats(cues, opts)
    longest = max((display_len(line) for c in cues for line in c.lines), default=0)
    parts = [f"**{st['cues']} cues**", f"{st['too_fast']} above {opts.max_cps:g} chars/s", f"longest line {longest} chars"]
    meta = meta or {}
    if meta.get("device"):
        dur, took = meta.get("duration", 0.0), meta.get("elapsed", 0.0)
        speed = f", {dur / took:.1f}x realtime" if took else ""
        parts.append(f"{meta.get('model')} on {meta['device']}{speed}")
    return " · ".join(parts)


def export_files(cues: list[Cue], stem: str, out_dir: Path, opts: dict) -> list[str]:
    """Write SRT, VTT and ASS for the current table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        str(F.write(cues, fmt, out_dir / f"{stem}.ar.{fmt}", **opts))
        for fmt in ("srt", "vtt", "ass")
    ]


def file_url(path: str | Path) -> str:
    return "/gradio_api/file=" + quote(Path(path).as_posix(), safe="/:")


def video_preview(path: str | Path | None, vtt: str | Path | None) -> str:
    """A plain <video> with a native subtitle track.

    Browsers render Arabic WebVTT correctly on their own. gr.Video is not used
    here because in Gradio 6 it does not show a video that carries subtitles.
    """
    if not path:
        return ""
    track = ""
    if vtt:
        track = f'<track kind="subtitles" label="Arabic" srclang="ar" src="{escape(file_url(vtt))}" default>'
    return (
        f'<video controls preload="metadata" src="{escape(file_url(path))}" '
        f'style="width:100%;max-height:360px;border-radius:10px;background:#000">{track}</video>'
    )


@lru_cache(maxsize=1)
def work_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="arsub-web-"))


def build_app():
    import gradio as gr

    has_gpu = cuda_device_count() > 0
    output_dir = work_dir()

    def make_subtitles(file, model, device, dialect, digits, strip, max_chars, translate, progress=gr.Progress()):
        if not file:
            raise gr.Error("Drop a video or audio file first.")
        path = Path(file if isinstance(file, str) else file.name)
        bopts = BuildOptions(max_chars=int(max_chars), digits=digits, remove_diacritics=bool(strip))
        topts = TranscribeOptions(model=model, device=device, dialect=dialect)
        progress(0.0, desc="Loading the model")
        try:
            result = run(
                path,
                out_dir=output_dir,
                formats=("vtt",),
                transcribe_opts=topts,
                build_opts=bopts,
                translate=bool(translate),
                progress=lambda f, m: progress(f, desc=m),
            )
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from exc
        preview = None if is_audio_only(path) else str(path)
        vtt = str(result.files["ar.vtt"])
        files = [str(p) for k, p in result.files.items() if k.startswith("en.")]
        return (
            cues_to_rows(result.cues),
            video_preview(preview, vtt),
            summary(result.cues, bopts, result.meta),
            str(path),
            result.meta,
            gr.update(value=files or None),
        )

    def _options(max_chars, digits, strip) -> BuildOptions:
        return BuildOptions(max_chars=int(max_chars), digits=digits, remove_diacritics=bool(strip))

    def apply_edits(rows, source, max_chars, digits, strip, meta):
        opts = _options(max_chars, digits, strip)
        try:
            cues = rows_to_cues(rows, opts)
        except ValueError as exc:
            raise gr.Error(f"Could not read a time in the table: {exc}") from exc
        stem = Path(source).stem if source else "subtitles"
        size = video_size(Path(source)) if source else (1920, 1080)
        files = export_files(cues, stem, output_dir, {"width": size[0], "height": size[1]})
        preview = gr.update()
        if source and not is_audio_only(source):
            preview = video_preview(source, files[1])
        return cues_to_rows(cues), preview, summary(cues, opts, meta), files

    def burn_now(rows, source, max_chars, digits, strip, meta, progress=gr.Progress()):
        if not source:
            raise gr.Error("Make subtitles first.")
        opts = _options(max_chars, digits, strip)
        cues = rows_to_cues(rows, opts)
        width, height = video_size(Path(source))
        ass = F.to_ass(cues, width=width, height=height)
        out = output_dir / f"{Path(source).stem}.subbed.mp4"
        try:
            burn_video(source, ass, out, size=(width, height), progress=lambda f, m: progress(f, desc=m))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc)) from exc
        return str(out)

    with gr.Blocks(title="arsub", analytics_enabled=False) as app:
        gr.Markdown(INTRO)
        source = gr.State(None)
        meta = gr.State({})
        with gr.Row():
            with gr.Column(scale=2):
                file = gr.File(label="Video or audio", file_count="single", type="filepath")
                go = gr.Button("Make subtitles", variant="primary")
                with gr.Accordion("Options", open=False):
                    model = gr.Dropdown(["auto", *MODELS], value="auto", label="Model")
                    device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label=f"Device ({'GPU found' if has_gpu else 'no GPU found'})")
                    dialect = gr.Dropdown(list(DIALECTS), value="none", label="Dialect hint")
                    digits = gr.Radio(list(DIGIT_STYLES), value="western", label="Digits")
                    strip = gr.Checkbox(label="Remove diacritics")
                    max_chars = gr.Slider(24, 60, value=42, step=1, label="Characters per line")
                    translate = gr.Checkbox(label="Also make English subtitles")
            with gr.Column(scale=3):
                player = gr.HTML(label="Preview")
                info = gr.Markdown()
        table = gr.Dataframe(
            headers=COLUMNS,
            datatype=["number", "str", "str", "str", "number"],
            column_count=5,
            interactive=True,
            wrap=True,
            label="Subtitles (edit any cell; ' / ' forces a line break)",
            elem_id="arsub-table",
        )
        with gr.Row():
            apply = gr.Button("Apply edits and export files")
            burn_btn = gr.Button("Burn into video")
        with gr.Row():
            downloads = gr.File(label="Subtitle files", file_count="multiple", interactive=False)
            burned = gr.Video(label="Burned video", interactive=False)

        go.click(
            make_subtitles,
            [file, model, device, dialect, digits, strip, max_chars, translate],
            [table, player, info, source, meta, downloads],
        )
        apply.click(apply_edits, [table, source, max_chars, digits, strip, meta], [table, player, info, downloads])
        burn_btn.click(burn_now, [table, source, max_chars, digits, strip, meta], [burned])
    return app


def launch_options(gr) -> dict:
    """Theme and CSS. System fonts only, so the browser never calls Google Fonts."""
    theme = gr.themes.Soft(
        primary_hue="teal",
        font=["Segoe UI", "system-ui", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    )
    return {"css": CSS, "theme": theme, "allowed_paths": [str(work_dir())]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arsub-web", description="Local web UI for arsub.")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="stay on 127.0.0.1 unless you really want other devices to connect")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    try:
        import gradio
    except ImportError:
        print("The web UI needs Gradio: pip install \"arsub[web]\"")
        return 1
    build_app().queue().launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        **launch_options(gradio),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
