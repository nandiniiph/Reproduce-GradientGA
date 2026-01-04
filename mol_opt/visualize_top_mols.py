# visualize_top_mols.py
import os
import yaml
from rdkit import Chem
from rdkit.Chem import Draw

# -----------------------------
# 1️⃣ SETTINGS
# -----------------------------
# Folder hasil run Gradient GA
results_folder = "results"  # ganti sesuai foldermu
method = "gradient_ga"
oracle = "QED"
seed = 0
top_n = 20  # jumlah molekul terbaik yang mau divisualisasi
output_file = f"top_{top_n}_mols_{oracle}.png"

# -----------------------------
# 2️⃣ LOAD DATA
# -----------------------------
# Cari file YAML atau PKL hasil run
file_name = f"{method}_{oracle}_{seed}.yaml"  # contoh format
file_path = os.path.join(results_folder, file_name)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

with open(file_path, "r") as f:
    data = yaml.safe_load(f)

# Data format: {SMILES1: [score, rank], SMILES2: [score, rank], ...}
# Ambil top-N berdasarkan score
top_mols = sorted(data.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
smiles_list = [smi for smi, _ in top_mols]
scores = [score[0] for _, score in top_mols]

# -----------------------------
# 3️⃣ CONVERT SMILES → RDKit Mol
# -----------------------------
mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]

# Hati-hati, hapus None jika SMILES invalid
mol_list_valid = []
scores_valid = []
for mol, score in zip(mol_list, scores):
    if mol is not None:
        mol_list_valid.append(mol)
        scores_valid.append(score)

# -----------------------------
# 4️⃣ PLOT GRID
# -----------------------------
img = Draw.MolsToGridImage(
    mol_list_valid,
    molsPerRow=5,
    subImgSize=(250, 250),
    legends=[f"{s:.3f}" for s in scores_valid]
)

# Tampilkan
img.show()

# Simpan ke file
img.save(output_file)
print(f"✅ Grid image tersimpan di: {output_file}")
