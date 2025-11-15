"""
Lightweight fuzzy inference system for ICU prioritization.

Outputs per-patient priority scores, survival estimates, and traceable rule
activations for explainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


def _triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _membership_sets(severity: float, stability: float, uncertainty: float):
    severity_low = _trapezoid(severity, 0.0, 0.0, 0.25, 0.45)
    severity_med = _triangular(severity, 0.3, 0.5, 0.7)
    severity_high = _trapezoid(severity, 0.6, 0.75, 1.0, 1.0)

    instability_high = _trapezoid(1 - stability, 0.4, 0.55, 1.0, 1.0)
    stability_high = _trapezoid(stability, 0.5, 0.65, 1.0, 1.0)

    certainty_low = _trapezoid(uncertainty, 0.12, 0.18, 0.25, 0.3)
    certainty_high = _trapezoid(1 - uncertainty, 0.5, 0.7, 1.0, 1.0)
    return {
        "severity_low": severity_low,
        "severity_med": severity_med,
        "severity_high": severity_high,
        "instability_high": instability_high,
        "stability_high": stability_high,
        "certainty_low": certainty_low,
        "certainty_high": certainty_high,
    }


@dataclass
class FuzzyScores:
    priority_score: float
    survival_score: float
    priority_band: str
    rule_trace: Dict[str, float]


def _evaluate_rules(row: pd.Series) -> FuzzyScores:
    severity = row["severity_score"]
    stability = row["stability_score"]
    uncertainty = row["uncertainty"]
    sets = _membership_sets(severity, stability, uncertainty)

    critical = max(
        min(sets["severity_high"], sets["instability_high"]),
        min(sets["severity_med"], sets["instability_high"], 1 - sets["certainty_high"]),
    )
    urgent = max(
        min(sets["severity_med"], sets["certainty_high"]),
        min(sets["severity_high"], sets["stability_high"], sets["certainty_high"]),
    )
    routine = max(
        min(sets["severity_low"], sets["stability_high"]),
        min(sets["severity_med"], sets["stability_high"], sets["certainty_high"]),
    )

    # defuzzify via weighted centroid
    weights = {"critical": 0.95, "urgent": 0.6, "routine": 0.25}
    numerator = critical * weights["critical"] + urgent * weights["urgent"] + routine * weights["routine"]
    denominator = max(critical + urgent + routine, 1e-6)
    priority_score = numerator / denominator

    survival_score = (
        0.6 * row["survival_proxy"]
        + 0.2 * (1 - sets["instability_high"])
        + 0.2 * sets["certainty_high"]
    )

    band = (
        "critical"
        if priority_score >= 0.75
        else "urgent"
        if priority_score >= 0.5
        else "routine"
    )
    trace = {
        "critical_rule": round(critical, 3),
        "urgent_rule": round(urgent, 3),
        "routine_rule": round(routine, 3),
        "severity_high": round(sets["severity_high"], 3),
        "instability_high": round(sets["instability_high"], 3),
        "certainty_high": round(sets["certainty_high"], 3),
    }
    return FuzzyScores(priority_score, survival_score, band, trace)


def compute_fuzzy_scores(patients: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict] = []
    for _, row in patients.iterrows():
        scores = _evaluate_rules(row)
        records.append(
            {
                "patient_id": row["patient_id"],
                "priority_score": round(scores.priority_score, 3),
                "survival_score": round(scores.survival_score, 3),
                "priority_band": scores.priority_band,
                "fuzzy_trace": scores.rule_trace,
            }
        )
    fuzzy_df = pd.DataFrame(records).set_index("patient_id")
    merged = patients.set_index("patient_id").join(fuzzy_df, how="left")
    merged = merged.reset_index()
    return merged

