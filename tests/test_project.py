from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "exports"
GT = ROOT / "data" / "ground_truth"


def test_required_dataset_sizes_after_pipeline():
    assert len(pd.read_csv(ROOT / "data/generated/programs.csv")) == 100
    assert len(pd.read_csv(ROOT / "data/generated/courses.csv")) == 1000
    assert len(pd.read_csv(ROOT / "data/generated/faculty.csv")) == 500
    assert len(pd.read_csv(ROOT / "data/generated/student_plans.csv")) == 20000


def test_ground_truth_separate_from_features():
    truth = pd.read_csv(GT / "future_actual_demand.csv")
    forecast = pd.read_csv(EXP / "demand_forecasts.csv")
    assert "actual_demand" not in forecast.columns
    assert "actual_demand" in truth.columns
    assert len(truth) == len(forecast) == 1000


def test_required_anomaly_scenarios_exist():
    df = pd.read_csv(EXP / "anomaly_scenarios.csv")
    expected = {"FACULTY_OUTSIDE_SKILL_SET", "WORKLOAD_EXCEEDED", "SECTION_CAPACITY_TOO_LOW", "PREREQUISITE_ORDER_VIOLATION", "DUPLICATE_SECTION_ASSIGNMENT"}
    assert expected.issubset(set(df.scenario))
    assert df.detected.all()
    assert (df.observed_detection == df.expected_detection).all()


def test_ids_and_forecast_outputs():
    courses = pd.read_csv(ROOT / "data/generated/courses.csv")
    sections = pd.read_csv(EXP / "proposed_sections.csv")
    allocations = pd.read_csv(EXP / "workload_allocations.csv")
    assert courses.course_id.iloc[0] == "CRS-000001"
    assert sections.section_id.iloc[0].startswith("SEC-2026-")
    assert allocations.workload_allocation_id.iloc[0].startswith("WLA-2026-")
    assert len(pd.read_csv(EXP / "demand_forecasts.csv")) == 1000
    assert json.loads((ROOT / "reports/evaluation.json").read_text())["MAE"] >= 0
