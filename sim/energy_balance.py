"""Monthly solar and battery energy balance (spec 6.3.1).

Symbols: g generation, L load, D direct use, C_in battery charge from solar,
B_out battery delivery, Z battery losses, X export, I grid import.
Identities: g = D + C_in + X;  L = D + B_out + I;  C_in = B_out + Z.
"""
from __future__ import annotations

import numpy as np


def self_consumption_fraction(load: np.ndarray, gen: np.ndarray, curve: dict, factor: float = 1.0) -> np.ndarray:
    """f_sc from the documented load-to-generation ratio curve (no battery)."""
    ratio = np.divide(load, gen, out=np.zeros_like(load, dtype=float), where=gen > 0)
    f = np.interp(ratio, curve["load_to_generation_ratio"], curve["f_sc"])
    return np.clip(f * factor, 0.0, 1.0)


def solar_battery_flows(
    g: np.ndarray,
    L: np.ndarray,
    days: np.ndarray,
    curve: dict,
    battery_kwh: float = 0.0,
    eta_rt: float = 0.9,
    cycles_per_day: float = 1.0,
    f_sc_factor: float = 1.0,
) -> dict[str, np.ndarray]:
    g = np.maximum(np.asarray(g, dtype=float), 0.0)
    L = np.maximum(np.asarray(L, dtype=float), 0.0)
    f_sc = self_consumption_fraction(L, g, curve, f_sc_factor)
    D = np.minimum(g * f_sc, L)
    S = g - D
    R = L - D
    if battery_kwh > 0 and eta_rt > 0:
        capacity = battery_kwh * days * cycles_per_day
        C_in = np.minimum(np.minimum(S, capacity), R / eta_rt)
    else:
        C_in = np.zeros_like(g)
    C_in = np.maximum(C_in, 0.0)
    B_out = eta_rt * C_in if battery_kwh > 0 else np.zeros_like(g)
    Z = C_in - B_out
    X = np.maximum(S - C_in, 0.0)
    I = np.maximum(R - B_out, 0.0)
    return {"g": g, "L": L, "D": D, "C_in": C_in, "B_out": B_out, "Z": Z, "X": X, "I": I, "f_sc": f_sc}


def gross_from_meters(import_kwh: float, generation_kwh: float, export_kwh: float, losses_kwh: float = 0.0) -> float:
    """Gross electricity L = I + g - X - Z from meter readings."""
    return import_kwh + generation_kwh - export_kwh - losses_kwh
