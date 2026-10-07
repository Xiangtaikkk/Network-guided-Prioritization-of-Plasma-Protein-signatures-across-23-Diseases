"""Create a small synthetic example (no real data): NPX matrix, limma-like table and protein universe.

The matrix has a latent block structure so that modules can be detected.  It is only meant to
test that the pipeline runs; it has no biological meaning.
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path(__file__).resolve().parent
n_samples, n_blocks, block_size = 120, 5, 24
proteins, blocks = [], []
for b in range(n_blocks):
    latent = rng.normal(size=(n_samples, 1))
    blocks.append(latent + 0.8 * rng.normal(size=(n_samples, block_size)))
    proteins += [f"P{b}_{k:02d}" for k in range(block_size)]
npx = pd.DataFrame(np.hstack(blocks), columns=proteins, index=[f"S{i:03d}" for i in range(n_samples)])
npx.to_csv(out / "example_npx.tsv", sep="\t")
limma = pd.DataFrame({"protein": proteins, "logFC": rng.normal(0, 0.6, len(proteins)), "adj.P.Val": rng.uniform(0.0, 1.0, len(proteins))})
dep_block = [p for p in proteins if p.startswith("P1_")]          # make block 1 enriched for DEPs
limma.loc[limma.protein.isin(dep_block), ["logFC", "adj.P.Val"]] = [1.5, 0.001]
limma.to_csv(out / "example_limma.tsv", sep="\t", index=False)
Path(out / "example_universe.txt").write_text("\n".join(proteins) + "\n")
print("example files written to", out)
