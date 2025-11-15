"""
Data ingestion and feature engineering utilities.

Transforms raw CSVs into feature-rich pandas DataFrames consumed by the
fuzzy and GA layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _scale(series: pd.Series, lower: float, upper: float) -> pd.Series:
    denom = max(upper - lower, 1e-6)
    return ((series - lower) / denom).clip(0.0, 1.0)


@dataclass
class PreparedData:
    patients: pd.DataFrame
    beds: pd.DataFrame


def load_data(
    patients_path: str | Path, beds_path: str | Path
) -> PreparedData:
    patients = pd.read_csv(patients_path)
    beds = pd.read_csv(beds_path)
    patients = _engineer_patient_features(patients)
    beds = _engineer_bed_features(beds)
    return PreparedData(patients=patients, beds=beds)


def _engineer_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severity_score"] = (
        0.6 * _scale(df["sofa_score"], 0, 20)
        + 0.4 * _scale(df["apache_ii_score"], 5, 45)
    )
    df["stability_score"] = (
        0.5 * (1 - _scale(df["lactate_mmol_l"], 0.4, 7.5))
        + 0.5 * _scale(df["mean_arterial_pressure"], 45, 110)
    )
    df["resource_demand"] = (
        0.5 * df["ventilator_probability"]
        + 0.2 * df["ventilator_need"]
        + 0.2 * df["dialysis_need"]
        + 0.1 * _scale(df["nurse_intensity"], 0.7, 2.2)
    )
    df["data_quality"] = (1 - df["uncertainty"]).clip(0.0, 1.0)
    df["survival_proxy"] = (1 - df["risk_score"]) * df["recommendation_score"]
    df["logistics_score"] = (
        0.6 * (1 - df["resource_demand"]) + 0.4 * df["stability_score"]
    )
    return df


def _engineer_bed_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ventilator_capacity"] = df["ventilator_available"].astype(int)
    df["dialysis_capacity"] = df["dialysis_ready"].astype(int)
    df["isolation_flag"] = df["isolation_room"].astype(int)
    df["nurse_capacity_norm"] = _scale(df["nurse_capacity"], 3.0, 6.0)
    return df


def build_feature_blocks(
    patients: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return semantically grouped feature views for XAI."""

    clinical = patients[
        [
            "patient_id",
            "age",
            "sofa_score",
            "apache_ii_score",
            "lactate_mmol_l",
            "mean_arterial_pressure",
            "severity_score",
            "stability_score",
        ]
    ]
    resource = patients[
        [
            "patient_id",
            "ventilator_need",
            "ventilator_probability",
            "dialysis_need",
            "nurse_intensity",
            "resource_demand",
        ]
    ]
    logistics = patients[
        [
            "patient_id",
            "specialty_need",
            "admission_type",
            "logistics_score",
            "length_of_stay_pred",
        ]
    ]
    data_quality = patients[
        [
            "patient_id",
            "uncertainty",
            "data_quality",
        ]
    ]
    return clinical, resource, logistics, data_quality

