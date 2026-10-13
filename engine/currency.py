"""Display currency. The engine works in GBP internally; users see and type rupees.

Rate and symbol live in data/reference/currency.json (placeholder rate; converted figures
are not local market prices).
"""
from __future__ import annotations

import json
from functools import lru_cache

from sim.parameters import REFERENCE_DIR


@lru_cache(maxsize=1)
def _cfg() -> dict:
    with open(REFERENCE_DIR / "currency.json", encoding="utf-8") as fh:
        return json.load(fh)


def rate() -> float:
    """Display units per GBP."""
    return float(_cfg()["per_gbp"])


def symbol() -> str:
    return _cfg()["symbol"]


def code() -> str:
    return _cfg()["code"]


def to_display(gbp: float | None) -> float | None:
    return None if gbp is None else gbp * rate()


def from_display(amount: float | None) -> float | None:
    return None if amount is None else amount / rate()


def group(n: float) -> str:
    """Indian digit grouping, e.g. 1234567 -> 12,34,567 (whole units)."""
    neg = n < 0
    s = str(int(round(abs(n))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if neg else "") + s


def fmt(amount: float) -> str:
    """Format an amount already in display currency: -₹1,200 / ₹12,34,567."""
    return ("-" if amount < 0 else "") + symbol() + group(abs(amount))


def money(gbp: float) -> str:
    """Convert a GBP amount and format it in display currency."""
    return fmt(to_display(gbp))


def per_kwh(gbp_per_kwh: float) -> float:
    """Unit price in display currency per kWh (2 decimals)."""
    return round(gbp_per_kwh * rate(), 2)
