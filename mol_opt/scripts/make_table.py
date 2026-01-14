import yaml
import numpy as np
import os
import csv
from collections import defaultdict

# =========================
# CONFIG (PAPER SETTING)
METHODS = {
    "gradient_ga": "Gradient GA",
    "graph_ga": "Graph GA",
    "smiles_ga": "SMILES GA",
    "mars": "MARS",
    "mimosa": "MIMOSA",
    "dst": "DST",
}

ORACLES = [
    "mestranol_similarity",
    "amlodipine_mpo",
    "perindopril_mpo",
    "deco_hop",
    "median1",
    "isomers_c9h10n2o2pf2cl",
]

SEEDS = [0, 1, 2, 3, 4]
MAX_CALLS = 2500
FREQ_LOG = 100
ROOT = "main"
OUT_CSV = "results/table2_2500.csv"
# =========================


def top_auc(buffer, top_n, finish=True):
    ordered = list(sorted(buffer.items(), key=lambda kv: kv[1][1]))
    prev = 0
    s = 0
    called = 0

    for idx in range(FREQ_LOG, min(len(buffer), MAX_CALLS), FREQ_LOG):
        temp = ordered[:idx]
        temp = sorted(temp, key=lambda kv: kv[1][0], reverse=True)[:top_n]
        topn = np.mean([x[1][0] for x in temp])
        s += FREQ_LOG * (topn + prev) / 2
        prev = topn
        called = idx

    temp = sorted(ordered, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    topn = np.mean([x[1][0] for x in temp])
    s += (len(buffer) - called) * (topn + prev) / 2

    if finish and len(buffer) < MAX_CALLS:
        s += (MAX_CALLS - len(buffer)) * topn

    return s / MAX_CALLS


def load_yaml(method, oracle, seed):
    path = f"{ROOT}/{method}/results/results_{method}_{oracle}_{seed}.yaml"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(buffer):
    items = sorted(buffer.items(), key=lambda kv: kv[1][0], reverse=True)
    scores = [x[1][0] for x in items]

    avg_top10 = np.mean(scores[:10])
    auc1 = top_auc(buffer, 1)
    auc10 = top_auc(buffer, 10)
    auc100 = top_auc(buffer, 100)

    return avg_top10, auc1, auc10, auc100


# =========================
# MAIN
os.makedirs("results", exist_ok=True)

print("\n=== TABLE 2 (2500 Oracle Calls) ===\n")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Oracle",
        "Method",
        "AvgTop10_mean", "AvgTop10_std",
        "AUC1_mean", "AUC1_std",
        "AUC10_mean", "AUC10_std",
        "AUC100_mean", "AUC100_std",
    ])

    for oracle in ORACLES:
        print(f"\n--- {oracle.upper()} ---")

        for method, label in METHODS.items():
            rows = []

            for seed in SEEDS:
                buffer = load_yaml(method, oracle, seed)
                if buffer is None:
                    continue
                rows.append(compute_metrics(buffer))

            if len(rows) == 0:
                continue

            rows = np.array(rows)
            mean = rows.mean(axis=0)
            std = rows.std(axis=0)

            # ===== TERMINAL (LaTeX-style) =====
            print(
                f"{label:12s} & "
                f"{mean[0]:.4f}±{std[0]:.4f} & "
                f"{mean[1]:.4f}±{std[1]:.4f} & "
                f"{mean[2]:.4f}±{std[2]:.4f} & "
                f"{mean[3]:.4f}±{std[3]:.4f} \\\\"
            )

            # ===== CSV =====
            writer.writerow([
                oracle,
                label,
                f"{mean[0]:.6f}", f"{std[0]:.6f}",
                f"{mean[1]:.6f}", f"{std[1]:.6f}",
                f"{mean[2]:.6f}", f"{std[2]:.6f}",
                f"{mean[3]:.6f}", f"{std[3]:.6f}",
            ])

print(f"\nSaved CSV → {OUT_CSV}\n")
