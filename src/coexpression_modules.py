"""
Disease-specific protein co-expression network and module construction
=======================================================================

Code accompanying Kong et al., "Network-Guided Prioritization of Plasma Protein
Signatures across 23 Diseases".

The module pipeline has three steps, applied to one disease at a time:

1. ``build_coexpression_network``  - Spearman correlation between all protein
   pairs, Benjamini-Hochberg (BH) correction across all protein pairs, and
   retention of positive associations with FDR < 0.05 as network edges.
2. ``detect_modules``              - the strongest ``top_fraction`` (default 5%)
   of the positive edges (ranked by correlation strength) define the network;
   modules are detected with the Leiden algorithm (modularity vertex
   partition) and modules with more than ``min_size`` proteins are kept.
3. ``find_key_modules``            - each module is tested for over-representation
   of the disease's differentially abundant proteins (DEPs) with a one-sided
   hypergeometric test against the shared protein universe; modules with
   P < 0.05 (unadjusted) are key modules.

Inputs are plain tables (see README.md); no data are distributed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from statsmodels.stats.multitest import multipletests

# ----------------------------------------------------------------------------
# Default parameters (those used for the results reported in the manuscript)
# ----------------------------------------------------------------------------
FDR_THRESHOLD = 0.05      # BH-adjusted P for network edges
TOP_FRACTION = 0.05       # strongest 5% of the positive edges define the network
MIN_MODULE_SIZE = 10      # keep modules with MORE than this number of proteins
SEED = 827                # random seed of the Leiden optimiser
KEY_MODULE_P = 0.05       # unadjusted hypergeometric P for key modules
DEP_FDR = 0.05            # DEP: adjusted P < 0.05 ...
DEP_ABS_LOGFC = 1.0       # ... and |log2 fold change| > 1


def build_coexpression_network(npx: pd.DataFrame, fdr: float = FDR_THRESHOLD,
                               positive_only: bool = True) -> pd.DataFrame:
    """Co-expression edge table from a samples x proteins NPX matrix.

    Spearman correlations for all protein pairs; P values are BH-corrected
    across all pairs of the disease network; edges with FDR < ``fdr`` (and
    positive correlation if ``positive_only``) are returned as a table with
    columns ``protein_1, protein_2, weight`` (Spearman rho) and ``padj``.
    """
    if npx.isna().any().any():
        raise ValueError("NPX matrix contains missing values; impute within the disease cohort first.")
    rho, pval = spearmanr(npx.values)
    cols = np.array(npx.columns)
    i, j = np.triu_indices(len(cols), k=1)
    edges = pd.DataFrame({"protein_1": cols[i], "protein_2": cols[j],
                          "weight": rho[i, j], "pval": pval[i, j]}).dropna()
    edges["padj"] = multipletests(edges["pval"], method="fdr_bh")[1]
    edges = edges[edges["padj"] < fdr]
    if positive_only:
        edges = edges[edges["weight"] > 0]
    return edges[["protein_1", "protein_2", "weight", "padj"]].reset_index(drop=True)


def detect_modules(edges: pd.DataFrame, top_fraction: float = TOP_FRACTION,
                   min_size: int = MIN_MODULE_SIZE, seed: int = SEED):
    """Leiden modules of the strongest positive edges.

    Returns ``(modules, modularity)`` where ``modules`` is a Series
    (index = protein, value = module id) restricted to modules with more than
    ``min_size`` proteins.
    """
    cols = {c.lower(): c for c in edges.columns}
    src = cols.get("protein_1") or cols.get("level_0")
    tgt = cols.get("protein_2") or cols.get("level_1")
    wcol = cols.get("weight")
    e = edges[[src, tgt, wcol]].rename(columns={src: "a", tgt: "b", wcol: "weight"})
    e = e[e["weight"] > 0]
    e = e[e["weight"] > e["weight"].quantile(1 - top_fraction)]
    graph = ig.Graph.TupleList(list(zip(e["a"], e["b"])), directed=False)
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(seed)
    partition = leidenalg.ModularityVertexPartition(graph)
    optimiser.optimise_partition(partition, n_iterations=-1)
    modules = pd.Series(partition.membership, index=graph.vs["name"])
    sizes = modules.value_counts()
    keep = sizes[sizes > min_size].index
    return modules[modules.isin(keep)], float(partition.modularity)


def find_key_modules(modules: pd.Series, deps, universe, p_threshold: float = KEY_MODULE_P) -> pd.DataFrame:
    """Hypergeometric over-representation of DEPs in every module.

    ``deps`` : iterable of differentially abundant proteins of the disease;
    ``universe`` : all proteins tested (e.g. the 1,154 shared proteins).
    """
    universe = set(universe)
    deps = set(deps) & universe
    rows = []
    for module_id in sorted(modules.unique()):
        members = set(modules.index[modules == module_id])
        overlap = len(members & deps)
        p = float(hypergeom.sf(overlap - 1, len(universe), len(deps), len(members))) if deps else 1.0
        rows.append({"module": int(module_id), "module_size": len(members), "n_deps": len(deps),
                     "overlap": overlap, "jaccard": overlap / max(len(members | deps), 1),
                     "p_value": p, "key_module": p < p_threshold})
    return pd.DataFrame(rows, columns=["module", "module_size", "n_deps", "overlap", "jaccard", "p_value", "key_module"])


def deps_from_limma(table: pd.DataFrame, protein_col: str = "protein", fdr_col: str = "adj.P.Val",
                    logfc_col: str = "logFC") -> set:
    """DEPs from a limma result table (adjusted P < 0.05 and |log2FC| > 1)."""
    keep = (table[fdr_col] < DEP_FDR) & (table[logfc_col].abs() > DEP_ABS_LOGFC)
    return set(table.loc[keep, protein_col])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npx", help="samples x proteins NPX matrix (tab-separated, first column = sample id)")
    src.add_argument("--edges", help="co-expression edge table (protein_1, protein_2, weight[, padj])")
    ap.add_argument("--deps", help="limma table of the disease (columns: protein, logFC, adj.P.Val)")
    ap.add_argument("--universe", help="text file with one protein per line (default: proteins in the NPX/edge table)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    ap.add_argument("--min-size", type=int, default=MIN_MODULE_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.npx:
        npx = pd.read_csv(a.npx, sep="\t", index_col=0)
        edges = build_coexpression_network(npx)
        edges.to_csv(out / "coexpression_edges.tsv", sep="\t", index=False)
        universe_default = list(npx.columns)
    else:
        edges = pd.read_csv(a.edges, sep="\t")
        universe_default = sorted(set(edges.iloc[:, 0]) | set(edges.iloc[:, 1]))
    modules, modularity = detect_modules(edges, a.top_fraction, a.min_size, a.seed)
    modules.rename("module").to_csv(out / "modules.tsv", sep="\t", header=True, index_label="protein")
    print(f"edges: {len(edges)} | modules (> {a.min_size} proteins): {modules.nunique()} | modularity: {modularity:.3f}")
    if a.deps:
        universe = [l.strip() for l in open(a.universe)] if a.universe else universe_default
        table = pd.read_csv(a.deps, sep="\t")
        key = find_key_modules(modules, deps_from_limma(table), universe)
        key.to_csv(out / "module_dep_enrichment.tsv", sep="\t", index=False)
        print(f"key modules (unadjusted P < {KEY_MODULE_P}): {int(key['key_module'].sum())} of {len(key)}")


if __name__ == "__main__":
    main()
