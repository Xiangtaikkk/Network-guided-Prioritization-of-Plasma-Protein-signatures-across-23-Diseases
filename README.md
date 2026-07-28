# Network-guided Prioritization of Plasma Protein Signatures across 23 Diseases

Code accompanying *Kong et al., "Network-Guided Prioritization of Plasma
Protein Signatures across 23 Diseases"*.

## Scientific purpose

We evaluate whether compact plasma protein panels discovered in the Human
Disease Blood Atlas (HDBA) transfer to an independent cohort, the UK Biobank
Pharma Proteomics Project (UKB-PPP), for two classification tasks per disease:
disease-versus-healthy and disease-versus-other-diseases, across 23 diseases.
A strictly separated three-stage design is used: candidate proteins are ranked
in HDBA (discovery), the model and panel size are selected on the UKB-PPP
training split, and performance is evaluated once on a held-out UKB-PPP test
split. An all-protein comparator restricted to the 1,154 HDBA/UKB-PPP shared
proteins provides the baseline reported in the manuscript.

This repository provides the **analysis code and the expected input data
format**. It does **not** include the HDBA or UKB-PPP data, which are
access-controlled; reproduction requires the user to obtain data access
independently (see "Input data requirements" below).

## Repository structure

```
.
├── README.md
├── LICENSE                     MIT
├── environment.yml             conda environment
├── requirements.txt            pip alternative
├── docs/pipeline_overview.md   three-stage design and repository scope
└── src/
    ├── config.py               centralised path resolution
    ├── hpa_panel_selection.py  Stage 1 - HDBA discovery
    ├── ukb_final_validation.py Stages 2-3 - UKB train/test (main results)
    ├── ukb_shared1154_comparator.py  1,154-protein comparator
    └── ukb_validation_repeated.py    20-seed robustness (supplementary)
```

## Software environment

- Python 3.12
- numpy, pandas, scipy, joblib, scikit-learn==1.6.1, xgboost, pyyaml

`scikit-learn` is pinned to 1.6.1 (the version used in the manuscript) for
reproducibility.

## Installation

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate proteomics-panel
```

Or pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Input data requirements

The pipeline reads access-controlled HDBA and UK Biobank data. Obtain the data
from the original providers and configure local paths:

Paths can alternatively be set via `HPA_UKB_<KEY>` environment variables.

The HDBA and UK Biobank data are **access-controlled and are not distributed
with this repository**. Obtain them from the original providers:

- **HDBA** — Human Protein Atlas, <https://www.proteinatlas.org/> (Olink
  Explore 1536).
- **UKB-PPP** — application-based access via
  <https://www.ukbiobank.ac.uk/using-the-resource/> (Olink Explore 3072;
  ICD-10 diagnoses from field 41270). Users must obtain their own UK Biobank
  data access to reproduce this analysis; no UK Biobank data or application
  details are included in this repository.

Expected files (set in `config.yaml`):

| Config key | File | Format |
|---|---|---|
| `HPA_DATA` | HDBA proteomics matrix | TSV, rows = samples (index), columns = protein symbols, values = NPX |
| `HPA_META` | HDBA metadata | TSV with a disease/group column and sample IDs |
| `UKB_DATA` | `ukb_data_disease.txt` | TSV, participants x protein NPX; used to read the shared-protein columns |
| `DATA_DIR` | dir with `ukb_meta_disease.txt`, `ukb_data_disease.txt`, `ukb_data_healthy.txt` | TSV; meta maps participants to disease labels, data files are participant x protein NPX |

Protein symbols must be consistent between HDBA and UKB-PPP so the shared
1,154-protein set can be derived. Missing NPX values are imputed inside the
pipeline using the training-data median only (no leakage). The 23 analysed
diseases are defined in-script (`DISEASES` in `src/hpa_panel_selection.py`);
no disease-list file is required.

## Reproducing the main analysis

Run from the `src/` directory (so `config.py` is importable):

```bash
cd src
python hpa_panel_selection.py        # Stage 1: HDBA discovery
python ukb_final_validation.py       # Stages 2-3: UKB train/test (main results)
python ukb_shared1154_comparator.py  # 1,154-protein comparator
python ukb_validation_repeated.py    # supplementary 20-seed robustness
```

`ukb_final_validation.py` must be run before `ukb_shared1154_comparator.py`
(the comparator reuses its results and helper functions).

## Expected outputs

Written under `RESULT_DIR/hpa_panel_v2/`:

| File | Content |
|---|---|
| `hpa_panel_final.csv` | final HDBA panel per disease x task |
| `hpa_all_models_cv.csv` | per-model HDBA cross-validation comparison |
| `hpa_protein_importance.csv` | best-model protein importance ranking |
| `ukb_final_results.csv` | main results: Test_AUC, AllProt_Test_AUC, metrics |
| `ukb_shared1154_comparator_results.csv` | 1,154-protein comparator metrics |
| `ukb_val_repeated_raw.csv`, `ukb_val_repeated_summary.csv` | 20-seed robustness |

## Data and code availability

This repository provides **code only**; no HDBA or UK Biobank data are
distributed. Users wishing to reproduce the analysis must obtain the
underlying data directly from the original providers.

- HDBA: Human Protein Atlas, <https://www.proteinatlas.org/>
- UK Biobank: application-based access, <https://www.ukbiobank.ac.uk/using-the-resource/>
- Code: this repository, released under the MIT License.

## Citation

If you use this code, please cite Kong et al. (manuscript in preparation).
