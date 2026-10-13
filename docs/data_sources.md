# Data sources and items to verify

Every numeric parameter is a **starting value or placeholder** until replaced from the cited source and recorded here with version and access date.

| Need | Source | File it replaces | Version / date | Licence checked |
|---|---|---|---|---|
| U-values, boiler efficiency, rating formula | SAP 10.2 / RdSAP 10 (BRE), TABULA/EPISCOPE | `data/reference/parameters.json`, `prices.json` (`rating_basis`) | TODO | TODO |
| Heating degree-days (base 15.5 °C) | Met Office / CIBSE / SAP Appendix U | `regions.json` | TODO | TODO |
| Solar yield by region | PVGIS | `regions.json` | TODO | TODO |
| Emission factors | DESNZ/DEFRA GHG conversion factors | `emission_factors.json` | TODO (2024 assumed) | OGL |
| Unit rates | Ofgem price cap | `prices.json` (`customer`) | TODO | OGL |
| Typical consumption | DESNZ NEED tables | `need_targets.json` | TODO | OGL |
| Monthly electricity shape | London Smart Meter dataset | `parameters.json` `electricity_monthly_shape` | TODO | TODO |
| Property mix | EPC open data (MHCLG) | `sampling_priors.json` via `gen.sampling.fit_from_epc` | TODO | TODO |
| Sales prices | HM Land Registry Price Paid | `data/raw/land_registry_epc.csv` | TODO | OGL |
| Retrofit costs and effects | EPC recommendations, Energy Saving Trust, scheme documents | `retrofit_catalogue.json` | TODO | TODO |

## Items most likely to be wrong (spec 16)

- SAP band thresholds, deflator (0.42 assumed from ECO4/SAP 2012) and fuel prices for SAP 10.2 / RdSAP 10. Validate by reproducing lodged EPC points from lodged costs and fitting the deflator per methodology version.
- U-value tables by age band and type.
- HDD base temperature and regional values.
- Heat pump SCOP range (2.6–3.5) and readiness threshold (HLP 2.0 W/m²K).
- Self-consumption curve and battery efficiency.
- Emission factor year and values; Ofgem unit rates.
- Retrofit costs; Boiler Upgrade Scheme grant amount and eligibility.
- Dataset licences and EPC ↔ Land Registry join feasibility.
