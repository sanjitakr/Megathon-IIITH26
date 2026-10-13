# Raw data (not committed)

Put licensed extracts here. Record the licence, version and access date in `docs/data_sources.md`.

| File | Source | Used by |
|---|---|---|
| `epc.csv` | EPC open data (MHCLG, registration needed), columns mapped to our vocabulary: region, property_type, age_band, wall_type, glazing, main_heating | `gen.sampling.fit_from_epc` |
| `land_registry_epc.csv` | HM Land Registry Price Paid joined to EPC: price, date, floor_area, property_type, region, age_band, epc_band | `ml.value_model --csv` |
| NEED tables | DESNZ NEED consumption medians by type and age | replace `data/reference/need_targets.json` |
| Smart meter subset | London Smart Meter dataset (monthly shape) | replace `electricity_monthly_shape` in `data/reference/parameters.json` |
