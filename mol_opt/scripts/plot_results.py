import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================
# CONFIG
METHODS = {
    "gradient_ga": "Gradient GA",
    "graph_ga": "Graph GA",
    "smiles_ga": "SMILES GA",
    "mars": "MARS",
    "mimosa": "MIMOSA",
    "dst": "DST",
}

TASK = "mestranol_similarity"
SEEDS = [0, 1, 2, 3, 4]
TOP_K = [10, 100]
MAX_CALLS = 2500

# Oracle calls that will be plotted (DISCRETE like paper)
PLOT_CALLS = np.array([
    100, 250, 500, 750, 1000, 1250,
    1500, 1750, 2000, 2250, 2500
])

ROOT = "main"
OUTDIR = "results"
OUTPNG = f"{OUTDIR}/mestranol_auc_all_methods.png"

# =========================
def load_seed(method, seed):
    path = f"{ROOT}/{method}/results/results_{method}_{TASK}_{seed}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    # (oracle_call, score)
    records = [(v[1], v[0]) for v in data.values()]
    records.sort()
    return records


def compute_auc(records, k):
    best_scores = []
    auc_curve = np.zeros(MAX_CALLS)

    idx = 0
    for call in range(1, MAX_CALLS + 1):
        while idx < len(records) and records[idx][0] <= call:
            best_scores.append(records[idx][1])
            idx += 1

        if best_scores:
            auc_curve[call - 1] = np.mean(
                sorted(best_scores, reverse=True)[:k]
            )
        else:
            auc_curve[call - 1] = 0.0

    # PAPER-LIKE: cumulative AUC (as used in many similarity search papers)
    return np.cumsum(auc_curve) / MAX_CALLS


# =========================
# PLOT
plt.figure(figsize=(13, 5))

for i, k in enumerate(TOP_K):
    plt.subplot(1, 2, i + 1)

    for method, label in METHODS.items():
        curves = []

        for seed in SEEDS:
            try:
                records = load_seed(method, seed)
                curves.append(compute_auc(records, k))
            except FileNotFoundError:
                pass

        if len(curves) == 0:
            continue

        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)

        x = PLOT_CALLS
        y = mean[PLOT_CALLS - 1]
        y_std = std[PLOT_CALLS - 1]

        # MAIN PLOT (DISCRETE POINTS)
        plt.plot(
            x, y,
            marker='o',
            linewidth=1.6,
            markersize=4,
            label=label
        )

        # SHADED STD
        plt.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.18
        )

    plt.xlabel("Oracle Calls")
    plt.ylabel(f"AUC Top-{k}")
    plt.title(f"Mestranol Similarity (Top-{k})")
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
os.makedirs(OUTDIR, exist_ok=True)
plt.savefig(OUTPNG, dpi=300)
plt.show()

print(f"Saved to {OUTPNG}")
