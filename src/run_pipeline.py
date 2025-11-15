"""
CLI entry point for ICU bed assignment optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data_loader import load_data
from .fuzzy_priority import compute_fuzzy_scores
from .ga_optimizer import BedAssignmentGA, GAConfig
from .xai import build_conflict_log, method_explanation, summarize_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid GA + Fuzzy ICU allocator")
    parser.add_argument("--patients", default="data/patients.csv")
    parser.add_argument("--beds", default="data/beds.csv")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--assignment_csv", default="out/assignments.csv")
    parser.add_argument("--report_json", default="out/report.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def ensure_output(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_pipeline(
    patients_path: str | Path,
    beds_path: str | Path,
    *,
    generations: int = 120,
    population: int = 80,
    verbose: bool = False,
    assignment_csv: str | Path | None = None,
    report_json: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    data = load_data(patients_path, beds_path)
    patients = compute_fuzzy_scores(data.patients)
    config = GAConfig(generations=generations, population_size=population)
    optimizer = BedAssignmentGA(patients, data.beds, config)
    best_solution, metrics = optimizer.run(verbose=verbose)
    assignments = optimizer.build_assignment_frame(best_solution)

    summary = summarize_assignments(assignments, patients)
    explanation = method_explanation(metrics, summary)
    conflicts = build_conflict_log(metrics, assignments)

    assignment_path = ensure_output(assignment_csv)
    if assignment_path:
        assignments.to_csv(assignment_path, index=False)

    report = {
        "optimization_score": metrics,
        "method_explanation": explanation,
        "conflict_resolution": conflicts,
        "assignment_csv": str(assignment_path) if assignment_path else None,
    }
    report_path = ensure_output(report_json)
    if report_path:
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return assignments, report


def main():
    args = parse_args()
    assignments, report = run_pipeline(
        args.patients,
        args.beds,
        generations=args.generations,
        population=args.population,
        verbose=args.verbose,
        assignment_csv=args.assignment_csv,
        report_json=args.report_json,
    )
    print("=== Optimization Summary ===")
    print(report["method_explanation"])
    if report.get("assignment_csv"):
        print(f"Assignments saved to {report['assignment_csv']}")
    if args.report_json:
        print(f"Report saved to {args.report_json}")
    conflicts = report.get("conflict_resolution", [])
    if conflicts:
        print("Conflicts handled:")
        for c in conflicts[:10]:
            print(f" - {c}")


if __name__ == "__main__":
    main()

