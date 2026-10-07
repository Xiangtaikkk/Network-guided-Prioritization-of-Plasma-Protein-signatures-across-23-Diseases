# Network-guided Prioritization of Plasma Protein Signatures across 23 Diseases

Code accompanying *Kong et al., "Network-Guided Prioritization of Plasma Protein
Signatures across 23 Diseases"*.

This repository contains two sets of scripts:

| Part | Content | Scripts |
|--|--|--|
| **1. Co-expression network and module construction** | disease-specific co-expression networks, Leiden modules and key-module identification | `src/coexpression_modules.py` |
| **2. Machine-learning panel discovery and validation** | HDBA discovery, UKB-PPP model/panel-size selection and held-out evaluation | `src/hpa_panel_selection.py`, `src/ukb_final_validation.py`, `src/ukb_shared1154_comparator.py`, `src/ukb_validation_repeated.py` |

Only the code of these two parts is provided. The HDBA and UKB-PPP data are not
distributed (see the data notes in each part).

---

# Part 1. Co-expression network and module construction

This part provides the code used to build disease-specific protein
co-expression networks, detect co-expression modules, and identify
disease-associated **key modules**, as described in the Methods section of the
manuscript ("Protein association network and disease-related key modules").

## What the code does

For one disease at a time, `src/coexpression_modules.py` performs three steps:

1. **Network construction** (`build_coexpression_network`)
   Spearman correlation between all protein pairs of the disease cohort (pairwise-complete
   observations if NPX values are missing); Benjamini-Hochberg correction across all protein
   pairs; positive associations with FDR < 0.05 are kept as edges. All proteins measured in
   the cohort are included.
2. **Module detection** (`detect_modules`)
   The strongest 5% of the positive edges (ranked by correlation strength) define
   the network. Modules are detected with the Leiden algorithm (modularity vertex
   partition), and modules with more than 10 proteins are retained.
3. **Key-module identification** (`find_key_modules`)
   Each module is tested for over-representation of the disease's differentially
   abundant proteins (DEPs; adjusted P < 0.05 and |log2 fold change| > 1 in the
   limma analysis against healthy controls) with a one-sided hypergeometric test
   against the shared protein universe. Modules with an unadjusted P < 0.05 are
   key modules.

## Data

No data are distributed with this code.

- HDBA data are publicly available through the Human Protein Atlas
  (https://www.proteinatlas.org).
- UKB-PPP data are available to approved researchers through application to the
  UK Biobank (https://www.ukbiobank.ac.uk/enable-your-research).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-modules.txt
```

Tested with Python 3.12 (see `requirements-modules.txt` for package versions).

## Usage

Input files (tab-separated):

| Option | File | Columns |
|--|--|--|
| `--npx` | NPX matrix of the disease cohort | first column = sample id, remaining columns = proteins |
| `--edges` | alternatively, an existing edge table | `protein_1`, `protein_2`, `weight`, optionally `padj` |
| `--deps` | limma result table of the disease | `protein`, `logFC`, `adj.P.Val` |
| `--universe` | list of tested proteins | one protein per line |

Command line (one disease at a time):

```bash
python src/coexpression_modules.py \
    --npx <disease_npx_matrix.tsv> \
    --deps <disease_limma_results.tsv> \
    --universe <shared_proteins.txt> \
    --out <output_directory>
```

To start from an existing edge table instead of an NPX matrix, replace `--npx` by
`--edges`. The defaults are the parameters used in the manuscript (see below).

Outputs written to `--out`:

- `coexpression_edges.tsv` (only with `--npx`) - FDR-significant positive edges
- `modules.tsv` - protein-to-module assignment
- `module_dep_enrichment.tsv` - module size, overlap with DEPs, Jaccard index,
  hypergeometric P value and key-module flag

Python interface:

```python
import pandas as pd
from src.coexpression_modules import (build_coexpression_network, detect_modules,
                                      find_key_modules, deps_from_limma)

edges = build_coexpression_network(npx)            # npx: samples x proteins
modules, modularity = detect_modules(edges)        # defaults of the manuscript
key = find_key_modules(modules, deps_from_limma(limma_table), universe)
```

## Parameters

| Parameter | Default | Meaning |
|--|--|--|
| `FDR_THRESHOLD` | 0.05 | BH-adjusted P for network edges |
| `TOP_FRACTION` | 0.05 | fraction of the strongest positive edges that define the network |
| `MIN_MODULE_SIZE` | 10 | modules with more than this number of proteins are kept |
| `SEED` | 827 | random seed of the Leiden optimiser |
| `KEY_MODULE_P` | 0.05 | unadjusted hypergeometric P for key modules |

## Reproducibility notes

- The Leiden optimiser is stochastic. The seed is fixed, but results can differ
  slightly between versions of `python-igraph` and `leidenalg`
  (versions used: see `requirements-modules.txt`).
- Starting from the co-expression edge tables used for the manuscript, the module
  detection and key-module steps of this code reproduce all 161 modules and the
  38 key modules reported in the manuscript.

## License

MIT (see `LICENSE`).

---

# Part 2. Machine-learning panel discovery and validation

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

This part of the repository provides the **analysis code and the expected
input data format**. It does **not** include the HDBA or UKB-PPP data, which are
access-controlled; reproduction requires the user to obtain data access
independently (see "Input data requirements" below).

## Repository structure

```
.
├── README.md
├── LICENSE                         MIT
├── requirements.txt                pip requirements of the machine-learning scripts
├── requirements-modules.txt        pip requirements of the module code (Part 1)
├── config.example.yaml             template for the local path configuration
├── pipeline_overview.md            three-stage design of the machine-learning scripts
└── src/
    ├── coexpression_modules.py     Part 1 - network and module construction
    ├── config.py                   centralised path resolution
    ├── hpa_panel_selection.py      Part 2, Stage 1 - HDBA discovery
    ├── ukb_final_validation.py     Part 2, Stages 2-3 - UKB train/test (main results)
    ├── ukb_shared1154_comparator.py  Part 2 - 1,154-protein comparator
    └── ukb_validation_repeated.py    Part 2 - 20-seed robustness (supplementary)
```

## Software environment

- Python 3.12
- numpy, pandas, scipy, joblib, scikit-learn==1.6.1, xgboost, pyyaml

`scikit-learn` is pinned to 1.6.1 (the version used in the manuscript) for
reproducibility.

## Installation

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

Copy `config.example.yaml` to `config.yaml` and edit the paths. Expected files:

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
