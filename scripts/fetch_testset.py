"""Download a small, openly licensed Arabic speech test set.

Three sources, each with a different kind of speech:

* FLEURS ar_eg (Google, CC BY 4.0): read Modern Standard Arabic, Egyptian speakers.
* Common Voice 18 Arabic (Mozilla, CC0), via the ungated mirror
  MohamedRashad/common-voice-18-arabic: short crowd-sourced read sentences.
* MASC (CC BY 4.0), via MohamedRashad/MASC-Arabic: speech cut from YouTube
  channels, many dialects, clean and noisy.

Only a few row groups / the start of an archive are read, so the whole thing is
about 100 MB instead of several GB. Clips land in data/testset/<source>/ and a
manifest.jsonl lists id, file, reference text, duration and license.

    python scripts/fetch_testset.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "testset"

FLEURS_TSV = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ar_eg/test.tsv"
FLEURS_TAR = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ar_eg/audio/test.tar.gz"
CV_PARQUET = "datasets/MohamedRashad/common-voice-18-arabic/data/test-00000-of-00001.parquet"
MASC_PARQUET = "datasets/MohamedRashad/MASC-Arabic/data/test-00000-of-00009.parquet"

LICENSES = {
    "fleurs": ("CC BY 4.0", "https://huggingface.co/datasets/google/fleurs"),
    "cv18": ("CC0 1.0", "https://huggingface.co/datasets/MohamedRashad/common-voice-18-arabic"),
    "masc": ("CC BY 4.0", "https://huggingface.co/datasets/MohamedRashad/MASC-Arabic"),
}


def retry(fn, tries: int = 8):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - flaky networks
            if i == tries - 1:
                raise
            print(f"  retrying after error: {exc}", file=sys.stderr)
            time.sleep(2 + 2 * i)


def fetch_fleurs(n: int) -> list[dict]:
    out_dir = OUT / "fleurs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = retry(lambda: urllib.request.urlopen(FLEURS_TSV, timeout=60).read().decode("utf-8"))
    refs: dict[str, tuple[str, str]] = {}
    for row in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) >= 4:
            refs[row[1]] = (row[0], row[2])
    seen_sentences: set[str] = set()
    items: list[dict] = []

    def stream() -> None:
        with urllib.request.urlopen(FLEURS_TAR, timeout=120) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    if len(items) >= n:
                        return
                    name = Path(member.name).name
                    if not member.isfile() or name not in refs:
                        continue
                    sentence_id, text = refs[name]
                    if sentence_id in seen_sentences:
                        continue  # one speaker per sentence keeps the set varied
                    data = tar.extractfile(member).read()  # type: ignore[union-attr]
                    path = out_dir / name
                    path.write_bytes(data)
                    seen_sentences.add(sentence_id)
                    items.append({"id": f"fleurs/{name}", "file": path.relative_to(ROOT).as_posix(), "text": text})

    retry(stream)
    return items


def fetch_parquet(repo_path: str, source: str, n: int, row_groups: list[int], text_col: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_dir = OUT / source
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    items: list[dict] = []
    per_group = -(-n // len(row_groups))

    def read_group(rg: int):
        with fs.open(repo_path, block_size=1 << 20) as f:
            return pq.ParquetFile(f).read_row_group(rg).to_pylist()

    for rg in row_groups:
        rows = retry(lambda: read_group(rg))
        taken = 0
        for i, row in enumerate(rows):
            text = (row.get(text_col) or "").strip()
            audio = row.get("audio") or {}
            data = audio.get("bytes")
            if not text or not data:
                continue
            if row.get("duration") and not (1.0 <= row["duration"] <= 30.0):
                continue
            orig = Path(audio.get("path") or f"{source}_{rg}_{i}.wav")
            name = f"{rg:03d}_{i:03d}{orig.suffix or '.wav'}"
            path = out_dir / name
            path.write_bytes(data)
            item = {"id": f"{source}/{rg}/{i}", "file": path.relative_to(ROOT).as_posix(), "text": text}
            if row.get("type"):
                item["type"] = row["type"]  # MASC labels clips clean or noisy
            items.append(item)
            taken += 1
            if taken >= per_group or len(items) >= n:
                break
        if len(items) >= n:
            break
    return items


def duration_of(path: Path) -> float:
    import av

    with av.open(str(path)) as c:
        if c.duration:
            return c.duration / 1_000_000
        s = c.streams.audio[0]
        return float(s.duration * s.time_base) if s.duration else 0.0


def write_index(manifest: list[dict]) -> Path:
    """eval/testset.csv: which clips were used and under what license (no audio, no text)."""
    path = ROOT / "eval" / "testset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator=chr(10))
        out.writerow(["id", "source", "type", "seconds", "license", "origin"])
        for it in manifest:
            out.writerow([it["id"], it["source"], it.get("type", ""), it["duration"], it["license"], it["origin"]])
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--masc", type=int, default=180, help="MASC clips are short, so take more of them")
    ap.add_argument("--sources", default="fleurs,cv18,masc")
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for source in sources:
        print(f"{source}: downloading clips")
        if source == "fleurs":
            items = fetch_fleurs(args.per_source)
        elif source == "cv18":
            items = fetch_parquet(CV_PARQUET, "cv18", args.per_source, [0, 35, 70], "sentence")
        elif source == "masc":
            items = fetch_parquet(MASC_PARQUET, "masc", args.masc, [0, 7, 14], "text")
        else:
            raise SystemExit(f"unknown source {source}")
        lic, url = LICENSES[source]
        for it in items:
            it.update(source=source, license=lic, origin=url, duration=round(duration_of(ROOT / it["file"]), 3))
        total = sum(it["duration"] for it in items)
        print(f"{source}: {len(items)} clips, {total / 60:.1f} min")
        manifest += items
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for it in manifest:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {OUT / 'manifest.jsonl'} ({len(manifest)} clips)")
    print(f"wrote {write_index(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
