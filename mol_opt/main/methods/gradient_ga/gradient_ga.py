from main.optimizer import BaseOptimizer
import numpy as np
import random
from rdkit import Chem

class Gradient_GA_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gradient_ga"

    def _optimize(self, oracle, config):
        """
        Implementasi Gradient GA sederhana:
        1. Inisialisasi populasi
        2. Iterasi sampai max_oracle_calls
        3. Evaluasi oracle
        4. Lakukan GA (mutation, crossover)
        5. Simpan best molecules
        """
        pop_size = config.get("population_size", 50)
        mutation_rate = config.get("mutation_rate", 0.2)
        max_steps = config.get("max_steps", 200)

        # Inisialisasi populasi random dari all_smiles
        population = random.sample(self.all_smiles, pop_size)
        best_score = -float("inf")
        best_mol = None

        for step in range(max_steps):
            # Hitung fitness
            scores = [oracle.score_smi(smi) for smi in population]

            # Update best
            idx = np.argmax(scores)
            if scores[idx] > best_score:
                best_score = scores[idx]
                best_mol = population[idx]

            # Simple mutation: ganti random atom dengan C/N/O
            new_population = []
            for smi in population:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                new_smi = self.mutate_smi(smi, mutation_rate)
                new_population.append(new_smi)
            population = new_population

            # Logging
            if step % self.oracle.freq_log == 0:
                self.log_intermediate()
        
        return best_mol, best_score

    def mutate_smi(self, smi, mutation_rate=0.2):
        """
        Simple mutation: ganti random atom
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        atoms = ["C", "N", "O", "F", "Cl", "Br"]
        mol_list = list(smi)
        for i in range(len(mol_list)):
            if random.random() < mutation_rate:
                mol_list[i] = random.choice(atoms)
        new_smi = "".join(mol_list)
        mol2 = Chem.MolFromSmiles(new_smi)
        if mol2 is None:
            return smi
        else:
            return Chem.MolToSmiles(mol2)
