# Green Home Score: method and sub-score mappings

`S = Σ w_k · s_k`, weights as fractions summing to 1, each sub-score `s_k` in 0–100, rounded to the nearest integer for reporting and classification. Code: `engine/score.py`. Every number below lives in `data/reference/score_config.json`.

| Component | Weight | Input | Mapping (a better state never scores lower) |
|---|---|---|---|
| Energy efficiency | 0.25 | Recorded EPC SAP points, or the recorded band anchored at its lower edge; if neither is known, our EPC-style estimate (flagged *estimated*) | Piecewise-linear on SAP points: 1→10, 21→25, 39→40, 55→55, 69→70, 81→85, 92→100. At band edges this equals the band table A 100, B 85, C 70, D 55, E 40, F 25, G 10. |
| Energy consumption intensity | 0.20 | Annualised, weather-normalised metered **gross** consumption per m² (electricity incl. solar used on site + gas/oil/LPG/solid) | 50→100, 100→80, 150→60, 250→30, 350→10, ≥450→0 kWh/m²/yr (lower is better; linear between anchors, clamped). Missing months are annualised from the months present with a seasonal shape, never treated as zero. |
| Insulation | 0.15 | Roof, wall, floor states | `0.4·roof + 0.4·wall + 0.2·floor`. Roof: none 0, <100 mm 35, 100–250 mm 70, >250 mm 100, flat uninsulated 10, flat insulated 80. Wall: solid 10, system-built 25, cavity unfilled 35, cavity filled 80, system insulated 90, solid insulated 100. Floor: suspended 30, solid 40, insulated 100. Elements a flat does not have get weight 0 and the rest are re-weighted. |
| Heating system | 0.15 | System and boiler age | Heat pump 100; gas boiler <10 yrs 65, 10–15 yrs 55, >15 yrs 45; LPG 55/47/40; oil 50/43/35; electric resistance 30; solid fuel/other 15. |
| Windows and glazing | 0.10 | Glazing type | Triple 100, modern double 80, pre-2002 double 55, mixed 40, secondary 35, single 20. |
| Renewable energy | 0.10 | Solar generation ÷ gross electricity demand | 0% → 0, rising linearly to 100 at 50% of demand. |
| Smart energy management | 0.05 | Controls | None 0, programmer 30, smart thermostat 60, smart TRVs / zoning 80, tariff-aware 100. |

**Classification:** 80–100 Highly efficient · 65–79 Good · 50–64 Moderate · 30–49 Poor · 0–29 Very poor.

**Brief example (P10001).** Sub-scores 40, 50, 40, 55, 20, 0, 30 give 10 + 10 + 6 + 8.25 + 2 + 0 + 1.5 = 37.75 → **38, Poor** (unit test `test_brief_example_is_38`).

## Why some inputs reach more than one component

- **Efficiency vs intensity.** Efficiency uses the recorded rating (standard occupancy, standard prices); intensity uses what the household actually used per m². They differ when behaviour or occupancy differs from the standard assumptions, which is information the score should carry.
- **Efficiency vs insulation, heating, glazing.** The rating summarises the whole home; the three components name the specific physical feature to fix. The overlap is visible in the contribution chart.
- **Solar.** Intensity uses gross consumption, so solar is not rewarded there; it is rewarded in the renewable component. The rating also credits solar (as SAP does) — the one remaining overlap.

## Explanation and safeguards

- Every component's `w_k × s_k`, its input and its basis (recorded, metered, estimated, modelled) is returned and shown in the dashboard.
- The score is marked *provisional* when an input was defaulted or no usage was given (Tier C).
- Current and projected scores use the same function; only the inputs differ (projected SAP = recorded SAP + simulated change; projected intensity = calibrated after-state).
- Poor-home detection lists the components losing the most weighted points as key issues. Homes are ranked by **improvement potential** `100 × (E_base − E_after)/E_base` (E = grid electricity import + gas, all feasible retrofits), not by total consumption, so large homes are not penalised for size alone. If E_base is zero or missing, the metric is reported as unavailable.
- Tests: monotonicity per component, contributions sum to the score, reproducibility, band-edge consistency (`tests/test_score.py`).
