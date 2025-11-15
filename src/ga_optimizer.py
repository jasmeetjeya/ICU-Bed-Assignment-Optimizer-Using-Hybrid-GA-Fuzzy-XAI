"""
Genetic Algorithm hybridized with fuzzy scores for ICU bed assignment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GAConfig:
    population_size: int = 80
    generations: int = 120
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    tournament_size: int = 3
    survival_weight: float = 0.55
    priority_weight: float = 0.35
    utilization_weight: float = 0.1


def _to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient="records")


class BedAssignmentGA:
    def __init__(
        self,
        patients: pd.DataFrame,
        beds: pd.DataFrame,
        config: GAConfig | None = None,
    ):
        self.patients_df = patients.reset_index(drop=True)
        self.beds_df = beds.reset_index(drop=True)
        self.config = config or GAConfig()
        self.bed_count = len(self.beds_df)
        self.population: List[np.ndarray] = []
        self.patient_lookup = {
            int(row.patient_id): row for row in self.patients_df.itertuples(index=False)
        }
        self.priority_pool = sorted(
            self.patient_lookup.keys(),
            key=lambda pid: (
                self.patient_lookup[pid].priority_score,
                self.patient_lookup[pid].survival_score,
            ),
            reverse=True,
        )
        self.total_nurse_capacity = float(self.beds_df["nurse_capacity"].sum())
        self.best_solution: np.ndarray | None = None
        self.best_fitness: float = float("-inf")
        self.best_metrics: Dict | None = None

    def run(self, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
        self.population = [self._initial_chromosome() for _ in range(self.config.population_size)]
        for gen in range(self.config.generations):
            fitnesses, metrics_list = zip(*(self._evaluate(ch) for ch in self.population))
            best_idx = int(np.argmax(fitnesses))
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = float(fitnesses[best_idx])
                self.best_solution = self.population[best_idx].copy()
                self.best_metrics = metrics_list[best_idx]
            if verbose and gen % 20 == 0:
                print(f"[GEN {gen}] fitness={self.best_fitness:.3f}")
            new_population: List[np.ndarray] = []
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_select(fitnesses)
                parent2 = self._tournament_select(fitnesses)
                child1, child2 = parent1.copy(), parent2.copy()
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                if random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population[: self.config.population_size]
        assert self.best_solution is not None and self.best_metrics is not None
        return self.best_solution, self.best_metrics

    def _initial_chromosome(self) -> np.ndarray:
        chrom = np.full(self.bed_count, -1, dtype=int)
        available = self.priority_pool.copy()
        for bed_idx in range(self.bed_count):
            bed = self.beds_df.iloc[bed_idx]
            candidate = self._select_candidate_for_bed(bed, available)
            if candidate is not None:
                chrom[bed_idx] = candidate
                available.remove(candidate)
        return chrom

    def _select_candidate_for_bed(self, bed_row, available: List[int]) -> int | None:
        for pid in available:
            patient = self.patient_lookup[pid]
            if patient.ventilator_need and not bed_row.ventilator_available:
                continue
            if patient.dialysis_need and not bed_row.dialysis_ready:
                continue
            return pid
        return None

    def _evaluate(self, chromosome: np.ndarray) -> Tuple[float, Dict]:
        assigned = set()
        survival_sum = 0.0
        priority_sum = 0.0
        occupancy = 0
        nurse_intensity_sum = 0.0
        constraint_penalty = 0.0
        mismatch_penalty = 0.0
        specialty_counts: Dict[str, int] = {}
        conflicts: List[str] = []

        for bed_idx, pid in enumerate(chromosome):
            if pid == -1:
                continue
            if pid in assigned:
                constraint_penalty += 1.5
                conflicts.append(f"duplicate assignment for patient {pid}")
                continue
            patient = self.patient_lookup.get(int(pid))
            if patient is None:
                constraint_penalty += 2.0
                continue
            bed = self.beds_df.iloc[bed_idx]
            assigned.add(pid)
            occupancy += 1
            survival_sum += float(patient.survival_score)
            priority_sum += float(patient.priority_score)
            nurse_intensity_sum += float(patient.nurse_intensity)

            if patient.specialty_need != bed.specialty:
                mismatch_penalty += 0.4
                conflicts.append(
                    f"specialty mismatch bed {bed.bed_id} ({bed.specialty}) -> patient {pid} ({patient.specialty_need})"
                )
            if patient.ventilator_need and not bed.ventilator_available:
                constraint_penalty += 1.0
                conflicts.append(f"ventilator shortage for patient {pid}")
            if patient.dialysis_need and not bed.dialysis_ready:
                constraint_penalty += 0.8
                conflicts.append(f"dialysis shortage for patient {pid}")

            specialty_counts[bed.specialty] = specialty_counts.get(bed.specialty, 0) + 1

        utilization = occupancy / max(self.bed_count, 1)
        survival_avg = survival_sum / max(occupancy, 1)
        priority_avg = priority_sum / max(occupancy, 1)

        nurse_ratio = nurse_intensity_sum / max(self.total_nurse_capacity, 1e-6)
        workload_penalty = max(0.0, nurse_ratio - 1.0) * 2.5

        if specialty_counts:
            counts = np.array(list(specialty_counts.values()), dtype=float)
            fairness_penalty = np.std(counts) / max(np.mean(counts), 1e-6) * 0.2
        else:
            fairness_penalty = 0.0

        fitness = (
            self.config.survival_weight * survival_avg
            + self.config.priority_weight * priority_avg
            + self.config.utilization_weight * utilization
            - (constraint_penalty + mismatch_penalty + workload_penalty + fairness_penalty)
        )

        metrics = {
            "survival_avg": survival_avg,
            "priority_avg": priority_avg,
            "utilization": utilization,
            "nurse_ratio": nurse_ratio,
            "constraint_penalty": constraint_penalty,
            "mismatch_penalty": mismatch_penalty,
            "workload_penalty": workload_penalty,
            "fairness_penalty": fairness_penalty,
            "conflicts": conflicts,
        }
        return fitness, metrics

    def _tournament_select(self, fitnesses: Tuple[float, ...]) -> np.ndarray:
        contenders = random.sample(range(len(self.population)), self.config.tournament_size)
        best_idx = max(contenders, key=lambda idx: fitnesses[idx])
        return self.population[best_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.random.rand(self.bed_count) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return self._repair(child1), self._repair(child2)

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        chrom = chromosome.copy()
        if random.random() < 0.5:
            i, j = random.sample(range(self.bed_count), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
        else:
            idx = random.randrange(self.bed_count)
            unused = [pid for pid in self.priority_pool if pid not in chrom]
            if unused:
                chrom[idx] = random.choice(unused)
        return self._repair(chrom)

    def _repair(self, chromosome: np.ndarray) -> np.ndarray:
        chrom = chromosome.copy()
        seen = set()
        available = [pid for pid in self.priority_pool if pid not in chrom]
        for idx, pid in enumerate(chrom):
            if pid == -1:
                continue
            if pid in seen:
                chrom[idx] = -1
            else:
                seen.add(pid)
        for idx, pid in enumerate(chrom):
            if chrom[idx] != -1:
                continue
            bed = self.beds_df.iloc[idx]
            replacement = self._select_candidate_for_bed(bed, available)
            if replacement is not None:
                chrom[idx] = replacement
                available.remove(replacement)
        return chrom

    def build_assignment_frame(self, chromosome: np.ndarray) -> pd.DataFrame:
        rows = []
        for bed_idx, pid in enumerate(chromosome):
            bed = self.beds_df.iloc[bed_idx]
            if pid == -1:
                rows.append(
                    {
                        "bed_id": bed.bed_id,
                        "specialty": bed.specialty,
                        "assigned_patient": None,
                        "reason": "left vacant due to constraint conflicts",
                    }
                )
                continue
            patient = self.patient_lookup[int(pid)]
            rows.append(
                {
                    "bed_id": bed.bed_id,
                    "specialty": bed.specialty,
                    "assigned_patient": int(pid),
                    "patient_specialty": patient.specialty_need,
                    "priority_score": patient.priority_score,
                    "survival_score": patient.survival_score,
                    "ventilator_need": int(patient.ventilator_need),
                    "dialysis_need": int(patient.dialysis_need),
                    "nurse_intensity": float(patient.nurse_intensity),
                    "reason": self._assignment_reason(bed, patient),
                }
            )
        return pd.DataFrame(rows)

    def _assignment_reason(self, bed, patient) -> str:
        parts = []
        if bed.specialty == patient.specialty_need:
            parts.append("specialty match")
        else:
            parts.append("no specialty match")
        if patient.ventilator_need:
            parts.append("ventilator provided" if bed.ventilator_available else "ventilator missing")
        if patient.dialysis_need:
            parts.append("dialysis ready" if bed.dialysis_ready else "dialysis missing")
        parts.append(f"priority {patient.priority_score:.2f}")
        return "; ".join(parts)

