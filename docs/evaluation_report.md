# Evaluation report

Generated 2026-10-11 07:45. Labels come from our own simulator: unless stated otherwise every metric below is **agreement with the reference simulator**, not real-world accuracy (spec 12.1).

## Baseline energy (agreement with the reference simulator, observed with noise and missing months)

| fuel / split | MAE kWh | MAPE % | R2 | 80% interval coverage | mean width kWh | baseline MAPE (type x age mean) |
|---|---|---|---|---|---|---|
| electricity_val | 685 | 21.0 | 0.877 | 0.800 | 2,233 | 54.1 |
| electricity_test | 709 | 20.9 | 0.901 | 0.797 | 2,342 | 53.4 |
| electricity_ood_test | 771 | 21.4 | 0.901 | 0.784 | 2,353 | 55.8 |
| gas_val | 1,820 | 16.5 | 0.785 | 0.800 | 5,573 | 63.6 |
| gas_test | 1,786 | 16.4 | 0.809 | 0.791 | 5,545 | 50.7 |
| gas_ood_test | 2,376 | 16.6 | 0.775 | 0.792 | 7,236 | 66.3 |

OOD = held-out region (Wales) and age band B (1900-1929), never seen in training.

### Ablations (test split)

| variant | MAE | MAPE % | R2 |
|---|---|---|---|
| electricity_physics_no_residual | 717 | 21.1 | 0.893 |
| electricity_no_physics_no_residual | 762 | 21.5 | 0.866 |
| electricity_physics_only | 704 | 20.2 | 0.900 |
| gas_no_physics_no_residual | 1,871 | 17.8 | 0.782 |
| gas_physics_only | 1,800 | 15.7 | 0.800 |
| full model (physics features + residual learning) | 709 | 20.9 | 0.901 |

Optional ensemble members: {"catboost": "not installed", "xgboost": "not installed"}

### Fairness: electricity MAPE % by group (test + OOD)

- **region**: east 21.7, east_midlands 22.0, london 19.5, north_east 22.1, north_west 22.8, northern_ireland 22.1, scotland 20.4, south_east 21.8, south_west 20.7, wales 21.7, west_midlands 20.3, yorkshire 20.9
- **property_type**: bungalow 21.2, detached 21.3, end_terrace 20.1, flat 21.0, semi 22.0, terraced 21.3
- **age_band**: A 22.2, B 21.2, C 20.7, D 20.9, E 20.7, F 21.0, G 22.5, H 21.3, I 20.5, J 23.8, K 21.0

## Shift test: trained on simulator A, tested on simulator B (parameters perturbed up to 20%)

| target | MAE | MAPE % | R2 |
|---|---|---|---|
| electricity_vs_version_A_clean | 683 | 21.3 | 0.914 |
| electricity_vs_version_B_clean | 688 | 23.1 | 0.913 |
| gas_vs_version_A_clean | 1,628 | 15.5 | 0.823 |
| gas_vs_version_B_clean | 2,119 | 15.6 | 0.757 |

This shows sensitivity to simulator misspecification, not which version is true.

## Feasibility (precision / recall on the rule labels, test split)

| retrofit | prevalence | classifier P/R | rules on defaults-filled inputs P/R | majority P/R |
|---|---|---|---|---|
| loft_insulation | 0.51 | 0.97 / 0.99 | 0.96 / 1.00 | 0.51 / 1.00 |
| cavity_wall_insulation | 0.19 | 0.99 / 0.92 | 1.00 / 0.92 | 0.00 / 0.00 |
| solid_wall_insulation | 0.18 | 0.99 / 0.95 | 1.00 / 0.95 | 0.00 / 0.00 |
| floor_insulation | 0.24 | 0.97 / 0.95 | 0.95 / 0.99 | 0.00 / 0.00 |
| double_glazing | 0.50 | 0.95 / 0.96 | 0.95 / 0.98 | 0.50 / 1.00 |
| draught_proofing | 0.44 | 0.97 / 0.96 | 0.95 / 0.99 | 0.00 / 0.00 |
| heat_pump | 0.47 | 0.98 / 0.98 | 0.96 / 0.99 | 0.00 / 0.00 |
| solar_pv | 0.43 | 0.98 / 1.00 | 0.98 / 1.00 | 0.00 / 0.00 |
| battery | 0.45 | 0.98 / 1.00 | 0.99 / 1.00 | 0.00 / 0.00 |
| smart_controls | 0.79 | 0.99 / 1.00 | 0.99 / 1.00 | 0.79 / 1.00 |

## Per-retrofit saving surrogates (MAE kWh vs simulator reference; baseline = fixed percentage like the brief's illustrative figures)

| retrofit | fuel | surrogate MAE | fixed-% baseline MAE | n |
|---|---|---|---|---|
| loft_insulation | d_elec | 32 | 75 | 1861 |
| loft_insulation | d_gas | 184 | 402 | 1861 |
| cavity_wall_insulation | d_elec | 59 | 179 | 683 |
| cavity_wall_insulation | d_gas | 415 | 905 | 683 |
| solid_wall_insulation | d_elec | 131 | 476 | 666 |
| solid_wall_insulation | d_gas | 846 | 1,259 | 666 |
| floor_insulation | d_elec | 28 | 95 | 875 |
| floor_insulation | d_gas | 167 | 230 | 875 |
| double_glazing | d_elec | 27 | 121 | 1834 |
| double_glazing | d_gas | 191 | 268 | 1834 |
| draught_proofing | d_elec | 7 | 39 | 1604 |
| draught_proofing | d_gas | 59 | 66 | 1604 |
| heat_pump | d_elec | 621 | 3,161 | 1721 |
| heat_pump | d_gas | 1,599 | 1,548 | 1721 |
| solar_pv | d_elec | 89 | 373 | 1568 |
| solar_pv | d_gas | 0 | 0 | 1568 |
| battery | d_elec | 103 | 511 | 1667 |
| battery | d_gas | 0 | 0 | 1667 |
| smart_controls | d_elec | 2 | 9 | 2918 |
| smart_controls | d_gas | 138 | 229 | 2918 |

## Retrofit ranking

NDCG@3 LambdaRank 0.982 vs heuristic (mean saving per unit of money) 0.912 over 3312 test homes. The engine itself ranks by exact simulated saving per unit cost within dependency tiers; LambdaRank is the ML comparison.

## Improvement potential

Spearman (model) 0.948; on homes with a known EPC band: model 0.950 vs EPC-band order 0.102. MAE 6.2 percentage points.

## Planner

Full enumeration vs rank-and-fill greedy (goal cut bills, budget ₹6,72,000, 60 homes): greedy mean optimality gap 3.63%, greedy suboptimal in 0.491 of homes. Median stage-1 7.5 ms, stage-2 re-rank 22.77 ms.

## Score

Monotonicity and reproducibility checks: loft pass, wall_type pass, glazing pass, controls pass, intensity pass, efficiency pass, reproducible pass

## Property value model

Basis: **synthetic_demo**. Test MAPE 14.93%, P10-P90 coverage 0.728, baseline (mean price by region and type) MAPE 26.93%.
The value model is the only component evaluated against real outcomes when trained on Land Registry data; it is observational.

## Held-out test predictions (full engine; files in data/predictions/)

3679 test homes run through the whole engine with their observed (noisy, partly missing) usage. Full-package savings vs simulator reference: electricity MAE 160 kWh, gas MAE 408 kWh (baseline: fixed % of baseline gas, MAE 3,103). Improvement-potential ranking Spearman 0.969 (EPC-band order 0.103). Poor/Very poor flagged: 31%. Abstained: 0.

## LLM layer

Questions 57; verifier pass rate 1.0; template fallback used 0.0. Manual audit: groundedness 0.702, factual accuracy 0.789, unsupported claims per sentence 0.0508 over 57 answers.

## Reporting rules

- Beating the simple baseline shows usefulness as a surrogate, not truth.
- Dashboard wording: estimated by a simulation model, not measured; savings are estimates, not guarantees.
