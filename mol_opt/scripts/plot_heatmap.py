import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "utils"))
from sascorer import calculateScore


# =========================
# CONFIG 
# =========================
METHODS = [
    "gradient_ga",
    "graph_ga",
    "smiles_ga",
    "mimosa",
    "mars",
    "dst",
]

METHOD_LABELS = [
    "Gradient GA",
    "Graph GA",
    "SMILES GA",
    "MIMOSA",
    "MARS",
    "DST",
]

ORACLES = [
    "perindopril_mpo",
    "mestranol_similarity",
    "median1",
    "isomers_c9h10n2o2pf2cl",
    "deco_hop",
    "amlodipine_mpo"
]

SEEDS = [0, 1, 2, 3, 4]
TOPK = 100

RESULT_ROOT = "main"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def compute_diversity(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    if len(fps) < 2:
        return np.nan

    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1 - sim)

    return np.mean(dists)


def load_topk_smiles(path, k=100):
    with open(path) as f:
        data = yaml.safe_load(f)

    # sort by oracle score (descending)
    items = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
    return [smi for smi, _ in items[:k]]


# =========================
# MAIN COMPUTATION
# =========================
sa_matrix = np.full((len(ORACLES), len(METHODS)), np.nan)
div_matrix = np.full((len(ORACLES), len(METHODS)), np.nan)

for i, oracle in enumerate(ORACLES):
    for j, method in enumerate(METHODS):
        sa_vals, div_vals = [], []

        for seed in SEEDS:
            path = f"{RESULT_ROOT}/{method}/results/results_{method}_{oracle}_{seed}.yaml"
            if not os.path.exists(path):
                continue

            smiles = load_topk_smiles(path, TOPK)

            # ---- SA ----
            sa_scores = [
                calculateScore(Chem.MolFromSmiles(s))
                for s in smiles
                if Chem.MolFromSmiles(s)
            ]

            # ---- Diversity ----
            diversity = compute_diversity(smiles)

            if len(sa_scores) > 0:
                sa_vals.append(np.mean(sa_scores))
            if not np.isnan(diversity):
                div_vals.append(diversity)

        if len(sa_vals) > 0:
            sa_matrix[i, j] = np.mean(sa_vals)
        if len(div_vals) > 0:
            div_matrix[i, j] = np.mean(div_vals)


# =========================
# PLOTTING 
# =========================
sns.set_style("white")
plt.rcParams.update({"font.size": 11})

# ---- FIGURE 5: SA ----
plt.figure(figsize=(7.2, 4.5))
sns.heatmap(
    sa_matrix,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    xticklabels=METHOD_LABELS,
    yticklabels=ORACLES,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "SA score (lower is better)"},
)
plt.title("Heatmap of synthetic accessibility (SA)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure5_sa_heatmap.png", dpi=300)
plt.close()

# ---- FIGURE 6: Diversity ----
plt.figure(figsize=(7.2, 4.5))
sns.heatmap(
    div_matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=METHOD_LABELS,
    yticklabels=ORACLES,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Diversity score (higher is better)"},
)
plt.title("Heatmap of diversity score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure6_diversity_heatmap.png", dpi=300)
plt.close()

print("✔ heatmaps saved")
