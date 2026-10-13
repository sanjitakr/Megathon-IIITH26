"""Customer memory: completed retrofits update the baseline, declined items are excluded,
and stated budget and disruption limits become defaults."""
from __future__ import annotations

import json
import re
from datetime import date

from sim.parameters import ROOT

MEMORY_DIR = ROOT / "data" / "memory"


def _path(customer_id: str):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", customer_id)[:80] or "anonymous"
    return MEMORY_DIR / f"{safe}.json"


def load(customer_id: str) -> dict:
    p = _path(customer_id)
    if not p.exists():
        return {"customer_id": customer_id, "completed": [], "declined": [], "budget": None, "disruption_tolerance": None, "history": []}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def save(mem: dict) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(mem["customer_id"]), "w", encoding="utf-8") as fh:
        json.dump(mem, fh, indent=2)


def record(customer_id: str, completed=None, declined=None, budget=None, disruption_tolerance=None) -> dict:
    mem = load(customer_id)
    for rid in completed or []:
        if rid not in mem["completed"]:
            mem["completed"].append(rid)
            mem["history"].append({"date": date.today().isoformat(), "event": "completed", "retrofit": rid})
        if rid in mem["declined"]:
            mem["declined"].remove(rid)
    for rid in declined or []:
        if rid not in mem["declined"]:
            mem["declined"].append(rid)
            mem["history"].append({"date": date.today().isoformat(), "event": "declined", "retrofit": rid})
    if budget is not None:
        mem["budget"] = budget
    if disruption_tolerance is not None:
        mem["disruption_tolerance"] = disruption_tolerance
    save(mem)
    return mem


def apply_to_input(inp, mem: dict):
    """Return a copy of the CustomerInput with memory merged in (explicit answers win)."""
    data = inp.model_dump()
    data["past_retrofits"] = sorted(set(data.get("past_retrofits") or []) | set(mem.get("completed", [])))
    data["declined"] = sorted(set(data.get("declined") or []) | set(mem.get("declined", [])))
    if data.get("budget") is None and mem.get("budget") is not None:
        data["budget"] = mem["budget"]
    if mem.get("disruption_tolerance") and data.get("disruption_tolerance") == "high":
        data["disruption_tolerance"] = mem["disruption_tolerance"]
    return type(inp)(**data)
