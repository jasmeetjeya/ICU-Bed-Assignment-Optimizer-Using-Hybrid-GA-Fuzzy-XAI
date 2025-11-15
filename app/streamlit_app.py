from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_pipeline import run_pipeline

DATA_DIR = Path("data")


@st.cache_data
def load_default_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _save_upload(file, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.getvalue())
    tmp.flush()
    return Path(tmp.name)


def main():
    st.set_page_config(page_title="ICU Bed Optimizer", layout="wide")
    st.title("ICU Bed Assignment Optimizer")
    st.caption("Hybrid GA + Fuzzy + XAI pipeline with hospital-ready dataset")

    col1, col2 = st.columns(2)
    with col1:
        patient_file = st.file_uploader("Upload patient CSV", type="csv")
        if patient_file is not None:
            patient_path = _save_upload(patient_file, ".csv")
        else:
            patient_path = DATA_DIR / "patients.csv"
            st.info("Using default `data/patients.csv`")
    with col2:
        bed_file = st.file_uploader("Upload bed CSV", type="csv")
        if bed_file is not None:
            bed_path = _save_upload(bed_file, ".csv")
        else:
            bed_path = DATA_DIR / "beds.csv"
            st.info("Using default `data/beds.csv`")

    st.sidebar.header("GA Settings")
    generations = st.sidebar.slider("Generations", min_value=20, max_value=200, value=80, step=10)
    population = st.sidebar.slider("Population", min_value=20, max_value=200, value=80, step=10)
    verbose = st.sidebar.checkbox("Verbose logs", value=False)

    if st.button("Run Optimization", type="primary"):
        with st.spinner("Running hybrid optimizer..."):
            assignments, report = run_pipeline(
                patient_path,
                bed_path,
                generations=generations,
                population=population,
                verbose=verbose,
                assignment_csv=None,
                report_json=None,
            )
        st.success("Optimization complete")
        metrics = report["optimization_score"]
        explanation = report["method_explanation"]
        conflicts = report["conflict_resolution"]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Avg survival score", f"{metrics['survival_avg']:.3f}")
        metric_cols[1].metric("Avg priority score", f"{metrics['priority_avg']:.3f}")
        metric_cols[2].metric("Bed utilization", f"{metrics['utilization']:.2f}")
        metric_cols[3].metric("Nurse load ratio", f"{metrics['nurse_ratio']:.2f}")

        st.subheader("Plain-language summary")
        assigned_count = int(assignments["assigned_patient"].count())
        total_beds = len(assignments)
        st.markdown(
            f"""
**Who got a bed?** {assigned_count} out of {total_beds} beds were filled. Empty slots only occur when no safe patient fits a bed’s equipment/staff limits.

**Survival score ({metrics['survival_avg']:.2f}):** higher is better. It blends fuzzy survival predictions and stability, so a score above 0.40 means we chose patients likely to benefit most from ICU care.

**Priority score ({metrics['priority_avg']:.2f}):** measures urgency (critical > urgent > routine). Values near 0.60 mean most beds went to high-risk patients first.

**Bed utilization ({metrics['utilization']*100:.0f}%):** shows how many beds are actively used. 100% = every bed filled safely.

**Nurse load ratio ({metrics['nurse_ratio']:.2f}):** compares total nursing demand from assigned patients to available nurse capacity. Below 1.0 keeps workloads sustainable.
"""
        )
        st.write("**Method explanation**")
        st.write(explanation)

        st.subheader("Assignments (detailed table)")
        st.caption("Columns show the bed, patient ID, medical needs, and quick reason codes.")
        st.dataframe(assignments, use_container_width=True)

        st.subheader("Assignments explained for non-technical readers")

        def describe_row(row):
            if pd.isna(row["assigned_patient"]):
                return (
                    f"Bed {row['bed_id']} ({row['specialty']}) kept empty because no patient "
                    "safely matched its specialty/equipment without overloading nurses."
                )
            patient_id = int(row["assigned_patient"])
            needs = []
            if row.get("ventilator_need", 0):
                needs.append("needs a ventilator to breathe")
            if row.get("dialysis_need", 0):
                needs.append("needs a dialysis machine for kidneys")
            need_text = ", ".join(needs) if needs else "needs routine monitoring only"
            specialty_note = (
                "is treated by the same type of specialists as this bed"
                if row["specialty"] == row.get("patient_specialty")
                else f"normally belongs in the {row.get('patient_specialty')} unit, but safely fits here"
            )
            urgency = (
                "critical"
                if row["priority_score"] >= 0.75
                else "urgent"
                if row["priority_score"] >= 0.5
                else "routine"
            )
            nurse_comment = (
                "keeps nurse workload comfortable"
                if row["nurse_intensity"] < 1.6
                else "pushes nurse workload higher than average"
            )
            human_reason = []
            reason = row["reason"]
            if "specialty match" in reason:
                human_reason.append("this bed already supports the right specialists")
            if "no specialty match" in reason:
                human_reason.append("no perfect specialty bed was free, so this was the safest alternative")
            if "ventilator provided" in reason:
                human_reason.append("bed has a ventilator ready for the patient")
            if "ventilator missing" in reason:
                human_reason.append("needs manual review because the bed lacks a ventilator")
            if "dialysis ready" in reason:
                human_reason.append("dialysis equipment is prepped for the patient")
            if "dialysis missing" in reason:
                human_reason.append("needs manual review because dialysis isn’t available")
            if not human_reason:
                human_reason.append("overall safety/benefit score was highest here")
            explanation_text = "; ".join(human_reason)
            return (
                f"Bed {row['bed_id']} ({row['specialty']}) hosts patient {patient_id}. "
                f"This patient is {urgency} and {specialty_note}. They {need_text}, and the pairing {nurse_comment}. "
                f"We picked this bed because {explanation_text}."
            )

        layman_rows = [describe_row(row) for _, row in assignments.iterrows()]
        explain_df = assignments[["bed_id", "assigned_patient"]].copy()
        explain_df["human_explanation"] = layman_rows
        for text in layman_rows[:10]:
            st.write(f"- {text}")
        if len(layman_rows) > 10:
            st.caption("Showing first 10 explanations; download CSV for full list.")

        st.subheader("Conflict log (what went wrong & why)")
        if conflicts:
            st.write(
                "Conflicts mean the optimizer accepted a soft rule violation (usually specialty mismatch) "
                "because no perfect option existed. Review them to decide if hospital policy allows the swap."
            )
            for c in conflicts:
                st.write(f"- {c}")
        else:
            st.write("No conflicts detected.")

        csv_bytes = assignments.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(report, indent=2).encode("utf-8")
        download_cols = st.columns(3)
        download_cols[0].download_button(
            "Download assignment CSV", csv_bytes, "assignments.csv", "text/csv"
        )
        download_cols[1].download_button(
            "Download report JSON", json_bytes, "report.json", "application/json"
        )
        explanation_bytes = explain_df.to_csv(index=False).encode("utf-8")
        download_cols[2].download_button(
            "Download layman explanations",
            explanation_bytes,
            "assignment_explanations.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()

