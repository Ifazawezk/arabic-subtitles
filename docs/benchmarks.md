# Benchmarks

Everything here comes from `scripts/evaluate.py` and is stored in
[`eval/results.json`](../eval/results.json), including the text each model wrote
for every clip, so any score can be audited or rescored.

## Machine

- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB, 591.74
- CPU: 13th Gen Intel(R) Core(TM) i9-13900H
- OS: Windows-11-10.0.26200-SP0
- Python 3.12.10, faster-whisper 1.2.1, CTranslate2 4.8.4, CUDA 12 wheels
- GPU runs use float16, CPU runs use int8

The laptop was in normal use during these runs, with other programs competing for
the CPU. Treat CPU speeds as a realistic busy-laptop figure, not a best case.

## Test set

| Source | Clips | Audio | Speech | License |
| --- | --- | --- | --- | --- |
| FLEURS ar_eg | 60 | 10.7 min | read Modern Standard Arabic | CC BY 4.0 |
| Common Voice 18 | 60 | 4.1 min | crowd-sourced, read | CC0 |
| MASC test split | 180 | 11.9 min | YouTube, many dialects, clean and noisy | CC BY 4.0 |

Every clip is listed with its license in [`eval/testset.csv`](../eval/testset.csv).
Reproduce it with `python scripts/fetch_testset.py`.

- **FLEURS** is Modern Standard Arabic read by speakers in Egypt.
- **Common Voice 18** is short sentences read by volunteers on their own devices.
- **MASC** is speech cut from over 700 YouTube channels. It is the most realistic
  set here and the only one with dialects. Its clips are labelled clean or noisy,
  not by dialect, so there is no per-dialect score yet.

Common Voice clips come from a public third-party mirror on Hugging Face, not from
Mozilla directly. Common Voice releases are CC0, but the mirror's own card only
says "cc", so check Mozilla's terms before reusing these clips.

## How scores are computed

- Models run with `language=ar`, beam size 5, VAD on and word timestamps on, the
  same settings as `arsub`.
- WER and CER are computed over all clips of a set together (total errors divided
  by total reference words), not averaged per clip, so short clips do not dominate.
- **Normalized** scores, the headline numbers, remove diacritics and punctuation,
  unify hamza forms, alef maqsura and teh marbuta, and convert digits. `ذهب إلى المدرسة`
  and `ذهب الى المدرسه` count as the same text, because that difference is a spelling
  convention, not a recognition error.
- **Raw** scores keep letters as written and remove only punctuation. They are in
  the table below for anyone who cares about exact spelling.

## Speed

Speed is measured on one long file: all clips joined with half a second of silence
between them. That exercises VAD, chunking and word timestamps the way a real video
does, and avoids per-call overhead. Model load and the first warm-up call are
excluded and reported separately. "x realtime" means seconds of audio per second of
wall time, so 10x means a 10 minute video takes 1 minute.

## Results

**Accuracy and speed on the GPU.** 300 clips, 26.7 minutes of audio. Lower WER and CER are better. Speed is how many seconds of audio are processed per second.

| Model | WER | CER | FLEURS | Common Voice | MASC | MASC clean | MASC noisy | Speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | 30.5% | 9.9% | 25.9% | 34.7% | 33.3% | 27.3% | 39.3% | 20.0x |
| medium | 20.2% | 6.0% | 14.8% | 21.9% | 24.2% | 19.9% | 28.6% | 11.2x |
| **large-v3-turbo** | **15.0%** | **4.8%** | 10.2% | 16.5% | 18.6% | 13.0% | 24.2% | **24.4x** |
| large-v3 | 13.4% | 4.4% | 8.1% | 13.8% | 17.7% | 14.5% | 20.9% | 5.5x |

**On the CPU.** 60 clips, compared with the GPU on the same clips, while the laptop was also doing other work.

| Model | CPU WER | GPU WER | CPU speed | GPU speed |
| --- | --- | --- | --- | --- |
| small | 26.1% | 26.4% | 3.5x | 20.0x |
| medium | 15.3% | 15.0% | 1.1x | 11.7x |
| large-v3-turbo | 10.8% | 10.8% | 1.3x | 24.5x |

**Do the options help?** Same 300 clips, `large-v3-turbo` on the GPU.

| Setting (large-v3-turbo) | WER | CER | MASC WER | Speed |
| --- | --- | --- | --- | --- |
| Default | 15.0% | 4.8% | 18.6% | 24.4x |
| `--batch-size 8` | 15.0% | 5.0% | 18.6% | 31.6x |
| `--dialect msa` | 14.9% | 4.7% | 18.5% | 16.2x |
| `--dialect egyptian` | 14.9% | 4.7% | 18.4% | 15.8x |

**A real film.** The 28 minute documentary from the gallery, narration over music, run through the same code as the `arsub` command. Wall time includes loading the model. The film has no reference transcript, so this table measures speed and subtitle timing, not accuracy.

| Model | Mode | Wall time | Speed | Words | Cues | Over 20 chars/s |
| --- | --- | --- | --- | --- | --- | --- |
| large-v3-turbo | GPU | 58s | 29.3x | 1596 | 164 | 0 |
| large-v3-turbo | GPU, batch 8 | 37s | 46.0x | 1593 | 163 | 0 |
| medium | GPU | 112s | 15.1x | 1589 | 170 | 0 |
| large-v3 | GPU | 126s | 13.5x | 1592 | 167 | 0 |
| small | CPU | 327s | 5.2x | 1604 | 171 | 0 |

## Raw scores and error types

| Model | Device | Clips | Raw WER | Normalized WER | Substitutions | Deletions | Insertions | Model load | First call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | cuda float16 | 300 | 41.6% | 30.5% | 720 | 86 | 28 | 1.2s | 0.8s |
| medium | cuda float16 | 300 | 33.5% | 20.2% | 461 | 58 | 35 | 3.3s | 1.4s |
| large-v3-turbo | cuda float16 | 300 | 28.6% | 15.0% | 337 | 55 | 19 | 3.7s | 1.1s |
| large-v3-turbo (batch 8) | cuda float16 | 300 | 28.6% | 15.0% | 333 | 59 | 19 | 2.4s | 1.3s |
| large-v3-turbo (egyptian prompt) | cuda float16 | 300 | 28.8% | 14.9% | 336 | 52 | 20 | 5.0s | 1.6s |
| large-v3-turbo (msa prompt) | cuda float16 | 300 | 28.8% | 14.9% | 331 | 54 | 22 | 3.2s | 1.5s |
| large-v3 | cuda float16 | 300 | 27.2% | 13.4% | 289 | 58 | 21 | 7.3s | 4.5s |
| small | cuda float16 | 60 | 35.2% | 26.4% | 160 | 12 | 4 | 1.5s | 0.8s |
| medium | cuda float16 | 60 | 25.8% | 15.0% | 91 | 6 | 3 | 3.4s | 1.5s |
| large-v3-turbo | cuda float16 | 60 | 20.1% | 10.8% | 62 | 7 | 3 | 3.5s | 1.1s |
| small | cpu int8 | 60 | 34.6% | 26.1% | 159 | 10 | 5 | 1.5s | 3.6s |
| medium | cpu int8 | 60 | 25.8% | 15.3% | 92 | 7 | 3 | 6.8s | 17.6s |
| large-v3-turbo | cpu int8 | 60 | 20.1% | 10.8% | 61 | 8 | 3 | 6.0s | 18.6s |

## Real video

[*History of Arabic Calligraphy*](https://commons.wikimedia.org/wiki/File:%D9%81%D9%8A%D9%84%D9%85_%D9%88%D8%AB%D8%A7%D8%A6%D9%82%D9%8A_%D8%AA%D8%A7%D8%B1%D9%8A%D8%AE_%D8%A7%D9%84%D8%AE%D8%B7_%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%8A-_%D9%85%D9%84%D8%A7%D9%85%D8%AD_%D9%85%D8%B5%D8%B1%D9%8A%D8%A9.webm) by Hamdy Khwaiter (CC BY 3.0), 28 minutes of narration over music. Wall time includes model load. There is no reference transcript, so this measures speed and timing only.

| Model | Mode | Wall time | Speed | Words | Cues | Over 20 chars/s |
| --- | --- | --- | --- | --- | --- | --- |
| large-v3-turbo | GPU | 58s | 29.3x | 1596 | 164 | 0 |
| large-v3-turbo | GPU, batch 8 | 37s | 46.0x | 1593 | 163 | 0 |
| medium | GPU | 112s | 15.1x | 1589 | 170 | 0 |
| large-v3 | GPU | 126s | 13.5x | 1592 | 167 | 0 |
| small | CPU | 327s | 5.2x | 1604 | 171 | 0 |

This film changed a default. An early version used a VAD minimum silence of 500 ms, and on this film that cut words from narration sitting on top of music, while the 300 short clips showed nothing wrong. The default is now 2000 ms, which keeps those words and also runs faster. Short clips alone would never have caught it.

## Limits of these numbers

- This is a small set: 300 clips and 26.7 minutes. Differences of a point or two between
  models may not be meaningful.
- WER is harsh on Arabic. A single wrong letter in a word that carries attached
  particles counts as a whole word wrong, which is why CER is shown beside it.
- MASC references come from existing transcripts of YouTube speech, and some are
  imperfect, so part of its error rate is noise in the reference itself.
- Numbers in references are often spelled out while Whisper writes digits. That
  counts as an error here, even when the subtitle is perfectly correct.
- FLEURS sentences are formal read text. A score on FLEURS says little about a
  casual Egyptian or Gulf conversation.
- CPU runs use a subset of 20 clips per source to keep them practical, so
  compare CPU rows with the GPU rows of the same subset.
