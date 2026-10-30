# Academic Course Demand Forecasting and Faculty Allocation System

A Python-based academic planning prototype that forecasts course demand,
proposes course sections, allocates qualified faculty, assigns rooms,
validates planning outputs, and exports results for review.

> **Project scope:** This prototype uses synthetic academic data.
> Forecast metrics are evaluated against separately stored synthetic
> future ground truth; they are not evidence of performance on real
> university enrollment.

## Key Features

-   Generates synthetic academic data for programs, courses, faculty,
    faculty skills, rooms, student plans, and historical enrollment.
-   Forecasts demand for 1,000 courses using a Scikit-learn
    `RandomForestRegressor`.
-   Evaluates predictions using Mean Absolute Error (MAE) and Root Mean
    Squared Error (RMSE).
-   Proposes course sections based on forecast demand and course
    capacity targets.
-   Allocates faculty based on availability, recorded course skills, and
    weekly workload limits.
-   Uses a greedy allocation heuristic that prioritizes courses with
    fewer eligible faculty members.
-   Assigns rooms that can accommodate section capacity and prevents
    duplicate room use in the same time slot.
-   Records unresolved allocation requirements as exceptions rather than
    assigning unqualified or over-capacity faculty.
-   Includes data validation, plan validation, predefined anomaly
    scenarios, and audit events.
-   Stores project data in SQLite and exports reports as CSV and JSON.
-   Includes automated tests with Pytest.

## Verified Run Results

The following results were produced using seed `20261022`:

  Metric                             Result
  ------------------------------ ----------
  Programs                              100
  Courses                             1,000
  Faculty members                       500
  Student course-plan records        20,000
  Course forecasts                    1,000
  Proposed sections                   1,327
  Sections assigned faculty           1,187
  Sections left unassigned              140
  Assignment rate                    89.45%
  Forecast MAE                        3.902
  Forecast RMSE                      4.7672
  Data validation issues                  0
  Plan validation issues                  0
  Predefined anomaly scenarios            5
  Automated tests                  4 passed

The 140 unassigned sections are recorded as exceptions. The allocation
algorithm is a greedy heuristic and does not guarantee a globally
optimal allocation. A section can remain unassigned when no currently
eligible faculty member has enough remaining workload capacity.

### Understanding the forecast metrics

-   **MAE (3.902):** the average absolute difference between predicted
    and synthetic ground-truth demand.
-   **RMSE (approximately 4.7672):** the square root of the mean squared
    prediction error, which gives larger errors more influence.

These metrics were calculated against synthetic future ground truth.
They should not be interpreted as validated accuracy on real
institutional enrollment.

## Technology Stack

-   **Python** --- application logic
-   **Pandas** --- data processing
-   **NumPy** --- numerical operations and synthetic data generation
-   **Scikit-learn** --- Random Forest regression
-   **SQLite** --- database storage
-   **Pytest** --- automated tests
-   **CSV / JSON** --- exported results

## Project Structure

The project is organized around the following main paths:

``` text
academic-planning-system/
├── src/
│   └── main.py
├── tests/
├── sql/
│   ├── schema.sql
│   └── schema_and_seed.sql
├── data/
│   ├── generated/
│   ├── ground_truth/
│   ├── exports/
│   └── academic_planning.sqlite
├── README.md
└── requirements.txt
```

Generated data, the SQLite database, and exports are created or updated
by the application. The exact contents of `tests/`, `sql/`, and `data/`
depend on the files included in the repository.

## Requirements

-   Windows, macOS, or Linux
-   Python 3.10 or a compatible version supported by the dependencies
-   `pip`

The commands below use **Windows Command Prompt (CMD)**.

## Setup on Windows CMD

### 1. Open the project directory

``` cmd
cd /d C:\D\academic-planning-system
```

Change the path if you cloned the repository to a different location.

### 2. Create a virtual environment

``` cmd
python -m venv .venv
```

### 3. Activate the virtual environment

``` cmd
.venv\Scripts\activate
```

### 4. Install dependencies

``` cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repository does not contain `requirements.txt`, create it from
the project's installed dependencies before publishing, or use the
dependency list supplied with the project.

## Run the Project

From the project root, with the virtual environment activated:

``` cmd
python -m src.main --seed 20261022
```

The seed makes the synthetic run reproducible for the same code and
dependency environment. A successful run prints a summary containing
dataset counts, assignment counts, forecast metrics, and the SQLite
database path.

The program writes its generated datasets and reports under the `data/`
directory.

## Run the Tests

``` cmd
python -m pytest -q
```

Verified result:

``` text
4 passed
```

The tests validate selected behaviors of the prototype. Passing tests do
not imply that every possible scheduling constraint or real-world
scenario is covered.

## Output Files

The application exports the following files under `data/exports/`:

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `demand_forecasts.csv`              Forecasted demand by course

  `forecast_summary.csv`              Forecast evaluation details

  `proposed_sections.csv`             Proposed sections, capacities, time
                                      slots, and room assignments

  `workload_allocations.csv`          Faculty allocation outcomes,
                                      including unassigned sections

  `exceptions.csv`                    Unresolved planning and
                                      resource-allocation exceptions

  `data_validation_issues.csv`        Data validation findings

  `plan_validation_issues.csv`        Planning validation findings

  `anomaly_scenarios.csv`             Results from predefined anomaly
                                      scenarios

  `audit_events.csv`                  Allocation and exception audit
                                      events

  `run_summary.json`                  Run configuration, record counts,
                                      forecast metrics, and ground-truth
                                      policy
  -----------------------------------------------------------------------

The SQLite database is stored at:

``` text
data/academic_planning.sqlite
```

## Validation and Anomaly Scenarios

The predefined anomaly scenarios demonstrate detection logic for:

1.  Faculty assigned outside their recorded skill set.
2.  Faculty workload exceeding its limit.
3.  Section capacity below demand.
4.  Prerequisite ordering violation.
5.  Duplicate section assignment.

These are deliberately constructed test scenarios. They demonstrate
selected checks and should not be interpreted as proof that all possible
anomalies are detected in operational data.

The current implementation does not enforce every possible real-world
constraint. In particular, room-type compatibility, all faculty
time-slot conflicts, and complete prerequisite scheduling should be
reviewed before any operational deployment.

## Reproducibility and Data Notes

-   The verified run used seed `20261022`.
-   Synthetic future ground truth is stored separately and used for
    evaluation, not as a model feature.
-   The generated data represents a test scenario and is not real
    student or faculty data.
-   Results can change if the code, library versions, or generated data
    logic changes.
-   Do not commit real student records, personal data, credentials, or
    other sensitive information to a public repository.

## Limitations

-   The project uses synthetic rather than institution-provided
    enrollment data.
-   Forecast performance has not been validated against real future
    enrollment.
-   Faculty allocation uses a greedy heuristic and may leave sections
    unassigned even when a different assignment order could improve the
    outcome.
-   The validation logic covers selected constraints, not every possible
    academic scheduling rule.
-   The predefined anomaly scenarios cover a limited set of failure
    cases.

## Possible Future Enhancements

-   Evaluate forecasts against privacy-protected historical enrollment
    data from an institution.
-   Compare the Random Forest model with alternative forecasting
    methods.
-   Add comprehensive prerequisite and faculty timetable conflict
    validation.
-   Enforce room-type requirements where relevant.
-   Improve allocation using reassignment or optimization methods if
    required.
-   Add a dashboard for demand, workload utilization, and exceptions.

## Project Report

A project report can accompany this repository as a PDF. It should be
kept consistent with the implementation and should clearly state that
the datasets and forecast evaluation ground truth are synthetic.

## License

No license is specified in this README. Add a `LICENSE` file before
publishing if you want to grant others explicit permission to use,
modify, and distribute the project. Choose a license that fits your
goals.

## Acknowledgement

This project demonstrates an end-to-end prototype for academic course
demand forecasting and resource planning using Python, machine learning,
validation rules, and structured reporting.
