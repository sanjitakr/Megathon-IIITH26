"""Number verifier (spec 11): every number in the reply must appear in the facts JSON the
LLM was given (allowing rounding and formatting), or in the user's own question."""
from __future__ import annotations

import math
import re

NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-\u2212]?(?:£|₹|Rs\.?\s?|INR\s?)?"
    r"(?:\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"  # Indian grouping, Western grouping, plain
)
CURRENCY_RE = re.compile(r"£|₹|Rs\.?\s?|INR\s?")
# Tokens whose digits are units or labels, not claims
IGNORE_RE = re.compile(r"CO2e?|CO₂e?|\bm2\b|m²|W/m2K|W/m²K|\bP(?:10|50|90)\b|\bkWh/m2\b|kWh/m²|\bSAP\s?10(?:\.2)?\b|\b24/7\b", re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"^\s*\d{1,2}[.)]\s", re.MULTILINE)


def flatten_numbers(obj, out: set | None = None) -> set[float]:
    out = set() if out is None else out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.add(float(obj))
        return out
    if isinstance(obj, str):
        for m in NUM_RE.findall(IGNORE_RE.sub(" ", obj)):
            v = _to_float(m)
            if v is not None:
                out.add(v)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            flatten_numbers(v, out)
    return out


def _to_float(token: str) -> float | None:
    t = CURRENCY_RE.sub("", token).replace(",", "").replace("−", "-")
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[str, float]]:
    cleaned = LIST_MARKER_RE.sub(" ", IGNORE_RE.sub(" ", text))
    out = []
    for m in NUM_RE.findall(cleaned):
        v = _to_float(m)
        if v is not None:
            out.append((m, v))
    return out


def _matches(written: str, n: float, v: float) -> bool:
    n, v = abs(n), abs(v)
    t = CURRENCY_RE.sub("", written).replace(",", "").lstrip("-−")
    decimals = len(t.split(".")[1]) if "." in t else 0
    if abs(n - v) <= 0.5 * 10 ** (-decimals) + 1e-9:
        return True
    if decimals == 0 and n >= 10:
        trailing = len(t) - len(t.rstrip("0"))
        if trailing and abs(n - v) <= 0.5 * 10 ** trailing + 1e-9:
            return True
    return False


def verify(text: str, facts, question: str = "") -> dict:
    allowed = flatten_numbers(facts)
    allowed |= {abs(x) for x in allowed}
    allowed |= {x * 100 for x in list(allowed) if abs(x) <= 1}  # fractions shown as percentages
    allowed |= flatten_numbers(question)
    unsupported = []
    for written, n in extract_numbers(text):
        if not any(_matches(written, n, v) for v in allowed):
            unsupported.append(written)
    return {"ok": not unsupported, "unsupported": unsupported, "n_numbers": len(extract_numbers(text))}
