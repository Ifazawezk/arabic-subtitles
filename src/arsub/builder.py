"""Turn timed words into readable subtitle cues.

The input is a flat list of words with start/end times (what Whisper gives
with word timestamps). The output is a list of cues that respect the usual
subtitle rules: a maximum line length, at most two lines, a reading speed
limit, minimum and maximum durations and a small gap between cues.

Arabic needs a few extra rules on top of that: line length is counted
without diacritics, a line should not end on a preposition or particle
("في", "من", "على", "اللي" ...), and compound names such as "عبد الله" or
"أبو بكر" stay on one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

from . import text as T


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]
    words: list[Word] = field(default_factory=list, repr=False)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def chars(self) -> int:
        return sum(T.display_len(line) for line in self.lines)

    @property
    def cps(self) -> float:
        return self.chars / self.duration if self.duration > 0 else float("inf")


@dataclass
class BuildOptions:
    max_chars: int = 42
    max_lines: int = 2
    max_cps: float = 20.0
    min_duration: float = 1.0
    max_duration: float = 7.0
    min_gap: float = 0.08
    pause: float = 0.7
    digits: str = "western"
    remove_diacritics: bool = False
    drop_hallucinations: bool = True


# A line should not end with one of these: they belong to the words after them.
DANGLING = frozenset(
    """
    في من الى إلى على عن مع حتى منذ مذ عند بين نحو لدى ضد خلال حول دون عبر تحت فوق
    امام أمام وراء بعد قبل لـ ب ك ل و ف
    او أو ثم لكن لكنّ بل ان أن إن انّ أنّ إنّ لان لأن كي لكي لم لن لا ما قد لقد سوف هل
    يا اذا إذا لو كأن كان الذي التي الذين اللذان اللتان اللواتي اللاتي اللي هذا هذه ذلك
    تلك هؤلاء اولئك أولئك كل بعض غير مثل ليس عشان علشان مش مو مب بدون زي اللى ده دي
    عبد ابو أبو ابي أبي بن ابن بنت ام أم آل ذو ذات
    حيث كيف لماذا متى اين أين عندما بينما كما لأنه لانه انه أنه إنه فيه منه عليه
    """.split()
)


def is_dangling(word: str) -> bool:
    """Would this word look stranded at the end of a line or cue?

    Also catches the same particles with a "و" or "ف" glued on: "وفي", "ومن", "فإن".
    """
    w = T.bare_word(word)
    if w in DANGLING:
        return True
    return len(w) > 2 and w[0] in "وف" and w[1:] in DANGLING

# Phrases Whisper writes on silence or music in Arabic audio. They look like
# subtitle credits, and nobody in the recording says them.
HALLUCINATIONS = (
    "ترجمة نانسي قنقر",
    "نانسي قنقر",
    "اشتركوا في القناة",
    "اشترك في القناة",
    "لا تنسوا الاشتراك في القناة",
    "المترجم للقناة",
    "Subtitles by the Amara.org community",
    "amara.org",
)


def _norm_for_match(s: str) -> str:
    s = T.strip_diacritics(T.strip_bidi_marks(s)).lower()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        s = s.replace(a, b)
    keep = [ch if ch.isalnum() or ch == "." else " " for ch in s]
    return " ".join("".join(keep).split())


_HALLUCINATIONS_NORM = tuple(sorted((_norm_for_match(h) for h in HALLUCINATIONS), key=len, reverse=True))


def is_hallucination(text: str) -> bool:
    """True when a cue is (almost) only known made-up credit lines.

    Known phrases are removed and whatever is left decides. Whisper often repeats
    a credit line, and the repeats get merged into a single cue, so a check for
    one phrase alone would let "ترجمة نانسي قنقر ترجمة نانسي قنقر" through.
    """
    norm = _norm_for_match(text)
    rest = norm
    for phrase in _HALLUCINATIONS_NORM:
        rest = rest.replace(phrase, " ")
    rest = " ".join(rest.split())
    return rest != norm and len(rest) <= 0.4 * len(norm)


def clean_words(words: list[Word], opts: BuildOptions) -> list[Word]:
    """Normalize every word and glue stray punctuation onto the word before it."""
    out: list[Word] = []
    for w in words:
        t = T.normalize_letters(T.strip_bidi_marks(w.text))
        if opts.remove_diacritics:
            t = T.strip_diacritics(t)
        t = T.convert_digits(t, opts.digits).strip()
        if not t:
            continue
        if not any(ch.isalnum() for ch in t) and out:
            prev = out[-1]
            out[-1] = replace(prev, text=prev.text + t, end=max(prev.end, w.end))
            continue
        out.append(Word(start=w.start, end=max(w.end, w.start), text=t, probability=w.probability))
    return out


def _line_len(words: list[str]) -> int:
    return T.display_len(" ".join(words))


def _break_cost(first: list[str], second: list[str], max_chars: int) -> float:
    """Cost of ending a line after ``first`` and starting the next with ``second``."""
    a, b = _line_len(first), _line_len(second)
    cost = abs(a - b)
    if a > b:
        cost += 0.5 * (a - b)  # slight preference for a bottom-heavy pyramid
    last = first[-1]
    if is_dangling(last):
        cost += max_chars
    if T.bare_word(last).isdigit():
        cost += max_chars / 2  # keep "5 دقائق" together
    if T.ends_sentence(last):
        cost -= max_chars
    elif T.ends_clause(last):
        cost -= max_chars / 2
    return cost


def split_lines(words: list[str], max_chars: int = 42, max_lines: int = 2) -> list[str]:
    """Split words into at most ``max_lines`` lines, only on word boundaries.

    A cue that fits on one line stays on one line. Otherwise every split point
    is scored (balance, punctuation, dangling particles) and the cheapest split
    whose lines all fit wins. If no split fits in ``max_lines`` lines, the same
    scoring is used with as many lines as needed.
    """
    words = [w for w in words if w]
    if not words:
        return []
    n = len(words)
    if _line_len(words) <= max_chars or max_lines <= 1 or n == 1:
        return [" ".join(words)]

    @lru_cache(maxsize=None)
    def best(i: int, k: int) -> tuple[float, tuple[int, ...]] | None:
        """Cheapest layout of words[i:] in at most k lines: (cost, cut positions)."""
        options: list[tuple[float, tuple[int, ...]]] = []
        if _line_len(words[i:]) <= max_chars:
            options.append((0.0, ()))
        if k > 1:
            for j in range(i + 1, n):
                line = words[i:j]
                if _line_len(line) > max_chars:
                    break
                rest = best(j, k - 1)
                if rest is not None:
                    # Each extra line costs a full line width, so fewer lines win.
                    step = 2 * max_chars + _break_cost(line, words[j:], max_chars)
                    options.append((step + rest[0], (j,) + rest[1]))
        return min(options) if options else None

    found = best(0, max_lines) or best(0, n)
    if found is None:  # a single word longer than a line
        return _greedy_wrap(words, max_chars)
    cuts = (0,) + found[1] + (n,)
    return [" ".join(words[a:b]) for a, b in zip(cuts, cuts[1:])]


def _greedy_wrap(words: list[str], max_chars: int) -> list[str]:
    lines: list[list[str]] = [[]]
    for w in words:
        if lines[-1] and _line_len(lines[-1] + [w]) > max_chars:
            lines.append([])
        lines[-1].append(w)
    return [" ".join(line) for line in lines]


def _fits(words: list[Word], opts: BuildOptions) -> bool:
    texts = [w.text for w in words]
    if _line_len(texts) <= opts.max_chars:
        return True
    lines = split_lines(texts, opts.max_chars, opts.max_lines)
    if len(lines) > opts.max_lines or any(T.display_len(line) > opts.max_chars for line in lines):
        return False
    # A cue whose only possible split leaves "في" or "من" hanging at a line end
    # counts as full; the grouping then cuts it one phrase earlier.
    return not any(is_dangling(line.split()[-1]) for line in lines[:-1])


def _best_cut(words: list[Word], next_word: Word, opts: BuildOptions) -> int:
    """Where to end a cue that just overflowed.

    ``words`` all fit; ``next_word`` did not. Look back over the second half of
    the cue for a nicer place to cut: after a sentence end, a comma or a pause,
    and not right after a dangling particle. Ties go to the fuller cue.
    """
    n = len(words)
    following = words + [next_word]
    candidates: list[tuple[float, int]] = []
    for i in range(max(1, (n + 1) // 2), n + 1):
        last = words[i - 1]
        score = 0.0
        if T.ends_sentence(last.text):
            score += 3
        elif T.ends_clause(last.text):
            score += 2
        gap = following[i].start - last.end
        score += min(gap / max(opts.pause, 1e-6), 1.5)
        if is_dangling(last.text):
            score -= 3
        score -= 0.15 * (n - i)
        candidates.append((score, i))
    return max(candidates)[1]


def group_words(words: list[Word], opts: BuildOptions) -> list[list[Word]]:
    """Greedy grouping of words into cues."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            too_long = w.end - cur[0].start > opts.max_duration
            sentence_done = T.ends_sentence(cur[-1].text) and _line_len([x.text for x in cur]) >= opts.max_chars * 0.5
            if gap >= opts.pause or too_long or sentence_done:
                groups.append(cur)
                cur = []
            elif not _fits(cur + [w], opts):
                cut = _best_cut(cur, w, opts)
                groups.append(cur[:cut])
                cur = cur[cut:]
        cur.append(w)
    if cur:
        groups.append(cur)
    return [g for g in groups if g]


def _merge_short(groups: list[list[Word]], opts: BuildOptions) -> list[list[Word]]:
    """Fold a tiny cue (one short word, a flash on screen) into a neighbour."""
    out: list[list[Word]] = []
    for g in groups:
        dur = g[-1].end - g[0].start
        tiny = dur < opts.min_duration / 2 and _line_len([w.text for w in g]) < 10
        if out and tiny:
            prev = out[-1]
            gap = g[0].start - prev[-1].end
            merged = prev + g
            if gap < opts.pause and merged[-1].end - merged[0].start <= opts.max_duration and _fits(merged, opts):
                out[-1] = merged
                continue
        out.append(g)
    return out


def fix_timing(cues: list[Cue], opts: BuildOptions) -> list[Cue]:
    """Enforce minimum duration, reading speed and a gap between cues.

    Cues are only ever extended into silence that follows them, so the text
    never shows up before the words are spoken.
    """
    for i, cue in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else None
        needed = max(opts.min_duration, cue.chars / opts.max_cps if opts.max_cps > 0 else 0.0)
        target_end = cue.start + needed
        if cue.end < target_end:
            limit = nxt - opts.min_gap if nxt is not None else target_end
            cue.end = max(cue.end, min(target_end, limit))
        cue.end = min(cue.end, cue.start + max(opts.max_duration, needed))
        if nxt is not None and nxt - cue.end < opts.min_gap:
            cue.end = max(cue.start + 0.2, nxt - opts.min_gap)
    return cues


def build_cues(words: list[Word], opts: BuildOptions | None = None) -> list[Cue]:
    """Words in, finished cues out."""
    opts = opts or BuildOptions()
    words = clean_words(sorted(words, key=lambda w: w.start), opts)
    groups = _merge_short(group_words(words, opts), opts)
    cues: list[Cue] = []
    for g in groups:
        lines = split_lines([w.text for w in g], opts.max_chars, opts.max_lines)
        lines = [T.fix_punctuation(line) for line in lines]
        lines = [line.lstrip("،,.؛; ") for line in lines]
        lines = [line for line in lines if line]
        if not lines:
            continue
        cue = Cue(start=round(g[0].start, 3), end=round(g[-1].end, 3), lines=lines, words=g)
        if opts.drop_hallucinations and is_hallucination(cue.text):
            continue
        cues.append(cue)
    return fix_timing(cues, opts)


def cue_from_text(start: float, end: float, text: str, opts: BuildOptions | None = None) -> Cue:
    """Build a cue from free text (used by the editor): clean it and re-split lines."""
    opts = opts or BuildOptions()
    words = T.clean_text(text.replace("\n", " "), opts.digits, opts.remove_diacritics).split()
    return Cue(start=start, end=end, lines=split_lines(words, opts.max_chars, opts.max_lines) or [""])


def stats(cues: list[Cue], opts: BuildOptions | None = None) -> dict:
    """Quick quality numbers for a list of cues."""
    opts = opts or BuildOptions()
    if not cues:
        return {"cues": 0, "too_fast": 0, "too_long_lines": 0, "max_cps": 0.0}
    return {
        "cues": len(cues),
        "too_fast": sum(1 for c in cues if c.cps > opts.max_cps + 1e-6),
        "too_long_lines": sum(1 for c in cues for line in c.lines if T.display_len(line) > opts.max_chars),
        "max_cps": round(max(c.cps for c in cues), 1),
    }
