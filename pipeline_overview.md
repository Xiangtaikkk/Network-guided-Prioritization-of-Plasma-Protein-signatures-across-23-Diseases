# Pipeline overview

The analysis is a strictly separated three-stage cross-cohort design.

```
Stage 1  HDBA discovery        src/hpa_panel_selection.py
  └─ 5-fold CV, 7 models, K = 1..20 sweep within HDBA only
  └─ Output: fixed top-20 candidate proteins per disease x task

Stage 2  UKB-train selection   src/ukb_final_validation.py (Stage 2)
  └─ 80% of UKB-PPP; RepeatedStratifiedKFold (5 x 4 = 20 folds)
  └─ Select best (model, K) from the HDBA top-20 candidates

Stage 3  UKB-test evaluation   src/ukb_final_validation.py (Stage 3)
  └─ Held-out 20% of UKB-PPP, evaluated once
  └─ Output: ukb_final_results.csv (Test_AUC + classification metrics)
```

Supporting analyses:

- `src/ukb_shared1154_comparator.py` — all-protein comparator restricted to
  the 1,154 shared proteins (the comparator reported in the manuscript).
- `src/ukb_validation_repeated.py` — supplementary robustness check over
  20 seeds with the panel fixed from Stage 1 (control-resampling sensitivity
  only; not a full re-run of the three-stage pipeline).
