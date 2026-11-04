# Contributing to arsub

Thanks for helping. Arabic speakers from every region are especially welcome,
because the hardest bugs here are the ones only a native reader notices.

## Set up

```bash
git clone https://github.com/Ifazawezk/arabic-subtitles
cd arabic-subtitles
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux and macOS
pip install -e ".[dev,web,eval]"
```

Add `gpu` to the extras if you have an NVIDIA card. Those wheels are over 1 GB.

## Run the tests

```bash
pytest
```

Unit tests never download a model. The transcriber is replaced by a fake, so
the suite finishes in seconds on any machine. One test runs a real ffmpeg burn
and skips itself when no ffmpeg with libass is available.

## The most useful contributions

**Bad line breaks.** If a subtitle splits in a place that reads wrong, open an
issue with the sentence and where it broke. Most fixes are one word added to
`DANGLING` in `src/arsub/builder.py` plus one test.

**Rendering bugs.** Name the player or TV, its version, and attach a short
`.srt` that shows the problem. Bidi behaviour differs a lot between players.

**Whisper hallucinations.** If the model invents a credit line on silence, add
the phrase to `HALLUCINATIONS` with a test. Only add text that could not
plausibly be real speech in that position.

**Dialect test data.** Openly licensed, transcribed dialect audio is what the
project needs most. Add it to `scripts/fetch_testset.py` with its license.

## Numbers

Any accuracy or speed figure in the README must come from
`scripts/evaluate.py` and be stored in `eval/results.json`. Do not edit results
by hand. If you run the benchmark on other hardware, open a pull request that
adds your numbers as a new section and name the machine.

## Before you open a pull request

- `pytest` passes.
- New behaviour has a test, including an Arabic example where it matters.
- Do not commit media, model weights or test audio. They are gitignored for a
  reason. Anything larger than a few MB should be fetched by a script.
- Keep source files free of literal invisible characters. Write bidi marks as
  `\u202b`, not as the raw character, so reviewers can see them.

## Commit messages

Short, imperative, and about what changed: `Keep compound names on one line`.
