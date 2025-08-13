# Here will be code for copy-paste

import numpy as np

import re
import unicodedata
import math
from typing import Any, List, Optional

_TILDE_SPLIT = re.compile(r"\s*~\s*")
_STRIP_CHARS = " \t\r\n\"'«»“”„‹›‐‑–—­\u00a0\u200b\u200c\u200d"

def split_erip_parts(value: Any, max_levels: int = 6, lowercase: bool = False) -> List[Optional[str]]:
    """
    Разбивает путь ERIP вида:
    'Система "Расчет" (ЕРИП) ~ Интернет-магазины/сервисы ~ A-Z Латинские домены ~ M ~ Mytesma.by'
    → ['Система "Расчет" (ЕРИП)', 'Интернет-магазины/сервисы', 'A-Z Латинские домены', 'M', 'Mytesma.by', None]

    - Чистит неразрывные пробелы/мягкие дефисы/кавычки по краям сегментов.
    - Терпимо к лишним пробелам вокруг тильд.
    - Паддит None до max_levels, либо обрезает длинные пути.
    """
    # Пустые/NaN
    if value is None:
        return [None] * max_levels
    if isinstance(value, float) and math.isnan(value):
        return [None] * max_levels

    # Нормализация юникода и пробелов
    s = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ")
    s = s.strip()

    if not s:
        return [None] * max_levels

    # Сплит по тильдам с игнорированием лишних пробелов
    raw_parts = _TILDE_SPLIT.split(s)

    parts: List[str] = []
    for p in raw_parts:
        p = p.strip(_STRIP_CHARS)
        if lowercase:
            p = p.lower()
        if p != "":
            parts.append(p)

    # Паддинг/обрезка
    if len(parts) < max_levels:
        parts += [None] * (max_levels - len(parts))
    else:
        parts = parts[:max_levels]

    return parts






 
"""
