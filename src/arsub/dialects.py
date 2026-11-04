"""Dialect presets.

Whisper has no switch for Arabic dialects; it only knows the language "ar".
What does work is the initial prompt: the decoder continues in the style of
the text it is primed with. A short prompt written in a dialect nudges the
spelling toward that dialect ("النهارده" instead of "اليوم"), and a prompt
with punctuation makes the model more likely to punctuate.

This is a bias, not a guarantee. The ``msa`` preset is the safe default for
news, lectures and formal speech.
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
