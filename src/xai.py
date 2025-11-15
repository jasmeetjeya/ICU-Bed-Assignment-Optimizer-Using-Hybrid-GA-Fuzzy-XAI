"""
Explainability helpers for ICU assignment decisions.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def summarize_assignments(assignments: pd.DataFrame, patients: pd.DataFrame) -> Dict:
    merged = assignments.merge(
        patients[
            [
                "patient_id",
                "severity_score",
                "stability_score",
                "resource_demand",
                "data_quality",
                "priority_band",
            ]
        ],
        how="left",
        left_on="assigned_patient",
        right_on="patient_id",
    )
    merged["severity_score"] = merged["severity_score"].fillna(0)
    feature_means = merged[
        ["severity_score", "stability_score", "resource_demand", "data_quality"]
    ].mean()
    feature_importance = {
        "severity": round(float(feature_means["severity_score"]), 3),
        "stability": round(float(feature_means["stability_score"]), 3),
        "resource_demand": round(float(feature_means["resource_demand"]), 3),
        "data_quality": round(float(feature_means["data_quality"]), 3),
    }
    priority_mix = merged["priority_band"].value_counts(normalize=True).to_dict()
    priority_mix = {k: round(float(v), 3) for k, v in priority_mix.items()}
    return {
        "feature_importance": feature_importance,
        "priority_mix": priority_mix,
    }


def build_conflict_log(metrics: Dict, assignments: pd.DataFrame) -> List[str]:
    conflicts = metrics.get("conflicts", []).copy()
    vacant_beds = assignments[assignments["assigned_patient"].isna()]
    for _, row in vacant_beds.iterrows():
        conflicts.append(
            f"bed {row['bed_id']} ({row['specialty']}) left vacant -> no safe candidate"
        )
    return conflicts


def method_explanation(metrics: Dict, summary: Dict) -> str:
    parts = [
        "Hybrid fuzzy-GA optimizer: fuzzy layer scores patients on severity, stability,",
        "and data certainty; GA searches bed allocations maximizing survival and",
        "priority while constraining nurse workload and equipment readiness.",
    ]
    parts.append(
        f"Average survival score {metrics['survival_avg']:.3f}, priority {metrics['priority_avg']:.3f}, utilization {metrics['utilization']:.2f}."
    )
    fi = summary["feature_importance"]
    parts.append(
        "Feature influence (avg normalized): "
        f"severity {fi['severity']}, stability {fi['stability']}, "
        f"resource demand {fi['resource_demand']}, data quality {fi['data_quality']}."
    )
    pmix = summary["priority_mix"]
    if pmix:
        dist = ", ".join(f"{band}:{share:.2f}" for band, share in pmix.items())
        parts.append(f"Priority mix across beds -> {dist}.")
    return " ".join(parts)

