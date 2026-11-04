"""Prompt presets.

Whisper has no switch for Arabic dialects; it only knows the language "ar".
These presets prime the decoder with a short sentence through its initial
prompt.

What they really do, measured on 300 clips and a real Egyptian recording: they
add punctuation and formal spelling ("لأن" instead of "لان", "طبعاً" instead of
"طبعا"). They do not move the text toward a dialect. Dialect words came through
just as well without a prompt, the choice of preset made almost no difference,
and overall WER did not change in a meaningful way. They also cost about a third
of the speed, so no preset is on by default.
"""

from __future__ import annotations

PROMPTS: dict[str, str] = {
    "msa": "مرحباً بكم. في هذا المقطع، نتحدث باللغة العربية الفصحى عن موضوعات مختلفة.",
    "egyptian": "أهلاً بيكم، إزيكم؟ النهارده هنتكلم عن حاجات كتير مهمة، يلا بينا.",
    "gulf": "هلا والله، شلونكم؟ اليوم بنسولف عن أشياء وايد حلوة، خلونا نبدأ.",
    "levantine": "أهلين فيكن، كيفكن؟ اليوم رح نحكي عن شغلات كتير مهمة، يلا نبلش.",
    "maghrebi": "السلام عليكم، لاباس عليكم؟ اليوم غادي نهضرو على بزاف ديال الحوايج.",
}

DIALECTS = ("none",) + tuple(PROMPTS)


def prompt_for(dialect: str | None) -> str | None:
    """The initial prompt for a dialect name, or ``None`` for no prompt."""
    if not dialect or dialect == "none":
        return None
    try:
        return PROMPTS[dialect]
    except KeyError:
        raise ValueError(f"unknown dialect {dialect!r}, expected one of {DIALECTS}") from None
