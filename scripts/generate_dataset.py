from __future__ import annotations

"""
ICU dataset generator anchored to published critical-care statistics.

- Age, SOFA, APACHE-II, comorbidity distributions informed by SCCM Fact &
  Figures (median ICU age ~62) and MIMIC-IV cohort summaries.
- Ventilator utilization baseline ~40-45% (SCCM 2022, eICU Collaborative DB).
- Diagnosis mix inspired by CDC ICU surveillance (sepsis, cardiac, neuro, trauma,
  respiratory failure, renal failure, post-op care, COVID-era ARDS cases).
"""

import csv
import random
from pathlib import Path


random.seed(2025)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SPECIALTIES = ["cardio", "neuro", "trauma", "pulmo", "renal", "general"]
ICU_TYPES = ["cardiac", "neuro", "surgical", "medical", "mixed"]
ADMISSION_TYPES = ["emergency", "urgent", "elective"]
DIAGNOSIS_GROUPS = [
    "sepsis",
    "cardiac_failure",
    "neuro_event",
    "poly_trauma",
    "ards",
    "renal_failure",
    "post_op",
    "covid_resp",
]
SEXES = ["F", "M"]

AGE_MEAN, AGE_SD = 62, 15
SOFA_MEAN, SOFA_SD = 7.5, 3.5
APACHE_MEAN, APACHE_SD = 22, 8
LOS_MEAN, LOS_SD = 6.4, 3.1
VENT_RATE_BASE = 0.42  # ~40-45% ICU vent utilization

DIAG_TO_SPECIALTY = {
    "sepsis": "general",
    "cardiac_failure": "cardio",
    "neuro_event": "neuro",
    "poly_trauma": "trauma",
    "ards": "pulmo",
    "covid_resp": "pulmo",
    "renal_failure": "renal",
    "post_op": "general",
}


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def generate_patients(n: int = 150):
    rows = []
    for pid in range(1, n + 1):
        age = int(clip(random.gauss(AGE_MEAN, AGE_SD), 18, 95))
        sex = random.choices(SEXES, weights=[0.42, 0.58])[0]
        weight = round(
            clip(random.gauss(78 if sex == "M" else 70, 12), 45, 150), 1
        )
        sofa = round(clip(random.gauss(SOFA_MEAN, SOFA_SD), 0, 20), 1)
        apache = int(clip(random.gauss(APACHE_MEAN, APACHE_SD), 5, 45))
        comorbidity_count = int(clip(random.gauss(2.1, 1.3), 0, 6))
        charlson = round(
            clip(random.gauss(4 + 0.6 * comorbidity_count, 1.2), 0, 12), 1
        )
        diagnosis = random.choices(
            DIAGNOSIS_GROUPS,
            weights=[0.22, 0.15, 0.12, 0.12, 0.11, 0.08, 0.12, 0.08],
        )[0]
        specialty_need = DIAG_TO_SPECIALTY[diagnosis]
        vitals_score = round(clip(random.gauss(6 + sofa / 4, 1.2), 0, 10), 1)
        lactate = round(clip(random.gauss(1.8 + 0.15 * sofa, 0.9), 0.4, 7.5), 1)
        map_value = round(clip(random.gauss(75 - 1.2 * sofa, 12), 45, 110), 1)
        admission_type = random.choices(
            ADMISSION_TYPES, weights=[0.6, 0.25, 0.15]
        )[0]
        length_of_stay_pred = round(
            clip(random.gauss(LOS_MEAN + 0.25 * (sofa - 7), LOS_SD), 1.2, 25), 1
        )
        nurse_intensity = round(
            clip(0.8 + 0.09 * sofa + random.uniform(-0.1, 0.18), 0.7, 2.2), 2
        )
        ventilator_prob = clip(
            VENT_RATE_BASE
            + 0.02 * (sofa - 7)
            + (0.09 if diagnosis in {"ards", "covid_resp"} else 0)
            + 0.015 * (charlson > 5),
            0.15,
            0.95,
        )
        ventilator_need = 1 if random.random() < ventilator_prob else 0
        dialysis_need = 1 if diagnosis == "renal_failure" and random.random() < 0.65 else 0
        risk_score = round(
            clip(
                0.3 + 0.028 * sofa + 0.012 * comorbidity_count + random.gauss(0, 0.04),
                0.2,
                0.98,
            ),
            3,
        )
        recommendation_score = round(
            clip(
                0.72 - 0.015 * sofa + 0.02 * (1 - comorbidity_count / 6)
                + random.gauss(0, 0.04),
                0.25,
                0.98,
            ),
            3,
        )
        uncertainty = round(
            clip(0.05 + 0.12 * random.random() + 0.01 * (admission_type == "emergency"), 0.05, 0.22),
            3,
        )
        rows.append(
            [
                pid,
                age,
                sex,
                weight,
                comorbidity_count,
                charlson,
                vitals_score,
                sofa,
                apache,
                diagnosis,
                specialty_need,
                admission_type,
                ventilator_need,
                round(ventilator_prob, 3),
                dialysis_need,
                lactate,
                map_value,
                length_of_stay_pred,
                risk_score,
                recommendation_score,
                nurse_intensity,
                uncertainty,
            ]
        )
    return rows


def generate_beds(n: int = 40):
    rows = []
    for bid in range(1, n + 1):
        icu_type = random.choices(ICU_TYPES, weights=[0.25, 0.15, 0.2, 0.2, 0.2])[0]
        specialty = random.choices(
            SPECIALTIES, weights=[0.2, 0.15, 0.18, 0.2, 0.1, 0.17]
        )[0]
        ventilator_available = 1 if random.random() < 0.65 else 0
        nurse_capacity = round(random.uniform(3.2, 5.8), 1)
        dialysis_ready = 1 if random.random() < 0.35 else 0
        isolation_room = 1 if random.random() < 0.3 else 0
        bedsides_monitors = 1
        rows.append(
            [
                bid,
                icu_type,
                specialty,
                ventilator_available,
                nurse_capacity,
                dialysis_ready,
                isolation_room,
                bedsides_monitors,
            ]
        )
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    patient_rows = generate_patients()
    bed_rows = generate_beds()
    write_csv(
        DATA_DIR / "patients.csv",
        [
            "patient_id",
            "age",
            "sex",
            "weight_kg",
            "comorbidity_count",
            "charlson_index",
            "vitals_score",
            "sofa_score",
            "apache_ii_score",
            "diagnosis_group",
            "specialty_need",
            "admission_type",
            "ventilator_need",
            "ventilator_probability",
            "dialysis_need",
            "lactate_mmol_l",
            "mean_arterial_pressure",
            "los_prediction_days",
            "risk_score",
            "recommendation_score",
            "nurse_intensity",
            "uncertainty",
        ],
        patient_rows,
    )
    write_csv(
        DATA_DIR / "beds.csv",
        [
            "bed_id",
            "icu_type",
            "specialty",
            "ventilator_available",
            "nurse_capacity",
            "dialysis_ready",
            "isolation_room",
            "advanced_monitoring",
        ],
        bed_rows,
    )
    print(f"Generated {len(patient_rows)} patients and {len(bed_rows)} beds.")


if __name__ == "__main__":
    main()

