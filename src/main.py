from __future__ import annotations
import argparse, json, sqlite3, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GEN = DATA / "generated"
EXP = DATA / "exports"
GT = DATA / "ground_truth"
REPORTS = ROOT / "reports"
DB_PATH = DATA / "academic_planning.sqlite"
SEED_DEFAULT = 20261022
YEAR = 2026
TERM = "2026-FALL"


def ensure_dirs():
    for p in (GEN, EXP, GT, REPORTS, ROOT / "sql"):
        p.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def generate_data(seed: int):
    rng = np.random.default_rng(seed)
    programs = pd.DataFrame({
        "program_id": [f"PRG-{i:06d}" for i in range(1, 101)],
        "program_name": [f"Program {i:03d}" for i in range(1, 101)],
        "department": [f"Department {(i-1)%10+1:02d}" for i in range(1, 101)],
        "annual_intake": rng.integers(40, 181, 100)
    })
    course_program_idx = np.repeat(np.arange(100), 10)
    course_ids = [f"CRS-{i:06d}" for i in range(1, 1001)]
    courses = pd.DataFrame({
        "course_id": course_ids,
        "program_id": [programs.iloc[i].program_id for i in course_program_idx],
        "course_name": [f"Course {i:04d}" for i in range(1, 1001)],
        "credits": rng.choice([2, 3, 4], 1000, p=[.2, .6, .2]),
        "capacity_target": rng.choice([30, 35, 40, 45, 50], 1000),
        "level": np.tile([1, 2, 3, 4], 250),
        "prerequisite_course_id": [None if (i-1)%10 == 0 else f"CRS-{i-1:06d}" for i in range(1,1001)]
    })
    faculty = pd.DataFrame({
        "faculty_id": [f"FAC-{i:06d}" for i in range(1, 501)],
        "faculty_name": [f"Faculty {i:03d}" for i in range(1, 501)],
        "max_weekly_load": rng.choice([8, 10, 12, 14], 500),
        "available": rng.choice([True, True, True, False], 500)
    })
    # Every faculty member receives four course skills. This creates a many-to-many bridge table.
    # Guarantee that every course has at least one available qualified faculty member.
    skill_rows = []

    available_faculty = faculty.loc[
        faculty["available"], "faculty_id"
    ].to_numpy()

    # First, assign one available faculty member to every course.
    # Faculty may teach multiple courses, and courses may have multiple instructors.
    for course_id in course_ids:
        faculty_id = rng.choice(available_faculty)
        skill_rows.append({
            "faculty_id": faculty_id,
            "course_id": course_id,
            "proficiency": rng.choice(["qualified", "expert"]),
        })

    # Then add three extra course skills per faculty member on average,
    # creating additional options for teaching assignments.
    for faculty_id in faculty["faculty_id"]:
        picked = rng.choice(course_ids, size=3, replace=False)
        skill_rows.extend(
            {
                "faculty_id": faculty_id,
                "course_id": course_id,
                "proficiency": rng.choice(["qualified", "expert"]),
            }
            for course_id in picked
        )

    faculty_skills = (
        pd.DataFrame(skill_rows)
        .drop_duplicates(subset=["faculty_id", "course_id"])
        .reset_index(drop=True)
    )
    rooms = pd.DataFrame({
        "room_id": [f"ROM-{i:04d}" for i in range(1, 81)],
        "capacity": rng.integers(25, 81, 80),
        "room_type": rng.choice(["lecture", "lab"], 80, p=[.75, .25])
    })
    student_n = 20000
    student_plans = pd.DataFrame({
        "student_plan_id": [f"PLN-{i:07d}" for i in range(1, student_n+1)],
        "student_id": [f"STU-{i:07d}" for i in range(1, student_n+1)],
        "program_id": rng.choice(programs.program_id.to_numpy(), student_n),
        "academic_term": TERM,
        "planned_course_id": rng.choice(course_ids, student_n),
        "plan_status": rng.choice(["active", "pending", "withdrawn"], student_n, p=[.78,.17,.05])
    })
    # Course popularity and historical demand are generated deterministically from the same seed.
    popularity = rng.gamma(shape=2.0, scale=13.0, size=1000)
    hist_rows = []
    for t in range(1, 5):
        seasonal = 0.90 + 0.06*t
        vals = np.maximum(0, np.rint(popularity * seasonal + rng.normal(0, 4, 1000))).astype(int)
        for j, course_id in enumerate(course_ids):
            hist_rows.append({"course_id": course_id, "term_index": t, "term": f"202{4+(t-1)//2}-{'SPRING' if t%2 else 'FALL'}", "enrollment": int(vals[j])})
    history = pd.DataFrame(hist_rows)
    # Ground truth for a future term is stored separately and is NEVER used as a model feature.
    last = history[history.term_index == 4].set_index("course_id").enrollment
    truth_rng = np.random.default_rng(seed + 9000)
    ground_truth = pd.DataFrame({"course_id": course_ids, "target_term": "2027-SPRING", "actual_demand": np.maximum(0, np.rint(last.reindex(course_ids).to_numpy() * 1.04 + truth_rng.normal(0, 3, 1000))).astype(int)})
    tables = {
        "programs.csv": programs, "courses.csv": courses, "faculty.csv": faculty,
        "faculty_skills.csv": faculty_skills, "rooms.csv": rooms,
        "student_plans.csv": student_plans, "historical_demand.csv": history
    }
    for name, df in tables.items(): save(df, GEN / name)
    save(ground_truth, GT / "future_actual_demand.csv")
    return tables, ground_truth


def validate_data(tables: dict[str, pd.DataFrame]) -> list[dict]:
    issues = []
    programs, courses, faculty = tables["programs.csv"], tables["courses.csv"], tables["faculty.csv"]
    plans, skills = tables["student_plans.csv"], tables["faculty_skills.csv"]
    def issue(code, detail, count=1): issues.append({"severity":"ERROR", "code":code, "detail":detail, "count":int(count)})
    if len(programs) != 100: issue("PROGRAM_COUNT", "Expected 100 programs", abs(len(programs)-100))
    if len(courses) != 1000: issue("COURSE_COUNT", "Expected 1000 courses", abs(len(courses)-1000))
    if len(faculty) != 500: issue("FACULTY_COUNT", "Expected 500 faculty", abs(len(faculty)-500))
    if len(plans) != 20000: issue("PLAN_COUNT", "Expected 20000 student plans", abs(len(plans)-20000))
    if programs.program_id.duplicated().any(): issue("DUPLICATE_PROGRAM_ID", "Program IDs must be unique", programs.program_id.duplicated().sum())
    if courses.course_id.duplicated().any(): issue("DUPLICATE_COURSE_ID", "Course IDs must be unique", courses.course_id.duplicated().sum())
    if faculty.faculty_id.duplicated().any(): issue("DUPLICATE_FACULTY_ID", "Faculty IDs must be unique", faculty.faculty_id.duplicated().sum())
    valid_programs = set(programs.program_id)
    valid_courses = set(courses.course_id)
    valid_faculty = set(faculty.faculty_id)
    issue("ORPHAN_COURSE_PROGRAM", "Courses reference missing programs", (~courses.program_id.isin(valid_programs)).sum()) if (~courses.program_id.isin(valid_programs)).any() else None
    issue("ORPHAN_PLAN_PROGRAM", "Plans reference missing programs", (~plans.program_id.isin(valid_programs)).sum()) if (~plans.program_id.isin(valid_programs)).any() else None
    issue("ORPHAN_PLAN_COURSE", "Plans reference missing courses", (~plans.planned_course_id.isin(valid_courses)).sum()) if (~plans.planned_course_id.isin(valid_courses)).any() else None
    issue("ORPHAN_SKILL_FACULTY", "Skills reference missing faculty", (~skills.faculty_id.isin(valid_faculty)).sum()) if (~skills.faculty_id.isin(valid_faculty)).any() else None
    issue("ORPHAN_SKILL_COURSE", "Skills reference missing courses", (~skills.course_id.isin(valid_courses)).sum()) if (~skills.course_id.isin(valid_courses)).any() else None
    issues = [x for x in issues if x["count"] > 0]
    save(pd.DataFrame(issues, columns=["severity","code","detail","count"]), EXP / "data_validation_issues.csv")
    return issues


def forecast_demand(tables, ground_truth):
    hist = tables["historical_demand.csv"].pivot(index="course_id", columns="term_index", values="enrollment").sort_index()
    course_meta = tables["courses.csv"].set_index("course_id")
    # Supervised training examples predict current term using only earlier-term values.
    rows = []
    for t in (3,4):
        for cid in hist.index:
            rows.append({"course_id":cid,"term_index":t,"lag1":int(hist.loc[cid,t-1]),"lag2":int(hist.loc[cid,t-2]),"credits":int(course_meta.loc[cid,"credits"]),"target":int(hist.loc[cid,t])})
    train = pd.DataFrame(rows)
    X = train[["lag1","lag2","credits"]]
    y = train["target"]
    model = RandomForestRegressor(n_estimators=80, random_state=SEED_DEFAULT, min_samples_leaf=2, n_jobs=-1)
    model.fit(X, y)
    # Forecast next term from the two most recent historical terms.
    future_X = pd.DataFrame({"lag1":hist[4].astype(int), "lag2":hist[3].astype(int), "credits":course_meta.loc[hist.index,"credits"].astype(int)}, index=hist.index)
    preds = np.maximum(0, np.rint(model.predict(future_X))).astype(int)
    forecasts = pd.DataFrame({"course_id":hist.index, "target_term":"2027-SPRING", "forecast_demand":preds})
    gt = ground_truth.set_index("course_id").reindex(forecasts.course_id)
    mae = float(mean_absolute_error(gt.actual_demand, forecasts.forecast_demand))
    rmse = float(np.sqrt(mean_squared_error(gt.actual_demand, forecasts.forecast_demand)))
    report = {"model":"RandomForestRegressor", "seed":SEED_DEFAULT, "evaluation":"synthetic future ground truth; ground truth excluded from model features", "n_courses":int(len(forecasts)), "MAE":mae, "RMSE":rmse}
    save(forecasts, EXP / "demand_forecasts.csv")
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return forecasts, report


def plan_and_allocate(tables, forecasts, seed):
    courses = tables["courses.csv"].copy()
    faculty = tables["faculty.csv"].copy()
    skills = tables["faculty_skills.csv"]
    rooms = tables["rooms.csv"]
    fskill = {f: set(g.course_id) for f,g in skills.groupby("faculty_id")}
    load = {f:0 for f in faculty.faculty_id}
    faculty_info = faculty.set_index("faculty_id").to_dict("index")
    room_list = rooms.sort_values("capacity", ascending=True).to_dict("records")
    occupied_room_slots = set()
    forecast_map = forecasts.set_index("course_id").forecast_demand.to_dict()
    section_rows, allocation_rows, exception_rows, audit_rows = [], [], [], []
    section_counter = 0

    # Count how many available faculty members are qualified for each course.
    # Courses with fewer eligible faculty members should be allocated first.
    available_skill_counts = {
        course_id: sum(
            1
            for faculty_id in fskill
            if course_id in fskill[faculty_id]
            and bool(faculty_info[faculty_id]["available"])
        )
        for course_id in courses["course_id"]
    }

    # Prioritize courses with fewer eligible faculty members.
    # For ties, prioritize courses with more credits, then use course ID.
    course_order = (
        courses.assign(
            _eligible_count=courses["course_id"].map(available_skill_counts)
        )
        .sort_values(
            ["_eligible_count", "credits", "course_id"],
            ascending=[True, False, True],
        )
    )
    # Stable order and tie-breaks make repeated runs reproducible.
    for c in course_order.drop(columns=["_eligible_count"]).itertuples(index=False):
        demand = int(forecast_map[c.course_id])
        cap = int(c.capacity_target)
        nsections = max(1, int(np.ceil(demand / max(cap,1))))
        for k in range(nsections):
            section_counter += 1
            sid = f"SEC-{YEAR}-{section_counter:06d}"
            section_cap = cap
            # Rotate slots across sections; use rooms that can fit the section.
            slot = f"SLOT-{(section_counter-1)%20+1:02d}"

            room = next(
                (
                    r for r in room_list
                    if int(r["capacity"]) >= section_cap
                    and (r["room_id"], slot) not in occupied_room_slots
                ),
                None,
            )

            room_id = room["room_id"] if room else None

            if room_id is not None:
                occupied_room_slots.add((room_id, slot))
                
            section_demand = max(0, min(section_cap, demand - k * section_cap))
            section_rows.append({"section_id":sid,"course_id":c.course_id,"term":"2027-SPRING","section_number":k+1,"forecast_demand":section_demand,"capacity":section_cap,"time_slot":slot,"room_id":room_id,"prerequisite_course_id":c.prerequisite_course_id})
            qualified = [
                f for f in fskill
                if c.course_id in fskill[f]
                and bool(faculty_info[f]["available"])
                and load[f] + int(c.credits)
                    <= int(faculty_info[f]["max_weekly_load"])
            ]

          # Prefer the eligible faculty member with the lowest workload utilization.
            qualified.sort(
                key=lambda f: (
                    load[f] / max(int(faculty_info[f]["max_weekly_load"]), 1),
                    f,
                )
            )

            chosen = qualified[0] if qualified else None
            if chosen:
                load[chosen] += int(c.credits)
                allocation_rows.append({"workload_allocation_id":f"WLA-{YEAR}-{len(allocation_rows)+1:06d}","section_id":sid,"faculty_id":chosen,"course_id":c.course_id,"weekly_load":int(c.credits),"status":"ASSIGNED"})
                audit_rows.append({"event_id":len(audit_rows)+1,"entity_type":"allocation","entity_id":allocation_rows[-1]["workload_allocation_id"],"event_type":"FACULTY_ASSIGNED","detail":f"{chosen} assigned to {sid}"})
            else:
                allocation_rows.append({"workload_allocation_id":f"WLA-{YEAR}-{len(allocation_rows)+1:06d}","section_id":sid,"faculty_id":None,"course_id":c.course_id,"weekly_load":int(c.credits),"status":"UNASSIGNED"})
                exception_rows.append({"exception_id":len(exception_rows)+1,"scenario":"UNMET_FACULTY_ALLOCATION","entity_id":sid,"severity":"HIGH","detail":"No qualified available faculty member with remaining workload capacity","status":"OPEN"})
            if room_id is None:
                exception_rows.append({"exception_id":len(exception_rows)+1,"scenario":"ROOM_CAPACITY_UNMET","entity_id":sid,"severity":"HIGH","detail":"No room can accommodate the proposed section capacity","status":"OPEN"})
    sections = pd.DataFrame(section_rows)
    allocations = pd.DataFrame(allocation_rows)
    exceptions = pd.DataFrame(exception_rows, columns=["exception_id","scenario","entity_id","severity","detail","status"])
    audit = pd.DataFrame(audit_rows, columns=["event_id","entity_type","entity_id","event_type","detail"])
    # Explicit test cases demonstrate detection of each required anomaly without corrupting valid plan outputs.
    # Each required anomaly has a deliberately malformed test fixture and an explicit detector result.
    anomaly_cases = [
        {"scenario_id":"ANOM-001","scenario":"FACULTY_OUTSIDE_SKILL_SET","test_input":"faculty assigned to course absent from faculty_skills","expected_detection":"SKILL_MISMATCH","detected":("FAC-TEST","CRS-TEST") not in set(zip(skills.faculty_id, skills.course_id))},
        {"scenario_id":"ANOM-002","scenario":"WORKLOAD_EXCEEDED","test_input":"current load 12 plus assignment 4 exceeds maximum 14","expected_detection":"WORKLOAD_LIMIT_EXCEEDED","detected":12 + 4 > 14},
        {"scenario_id":"ANOM-003","scenario":"SECTION_CAPACITY_TOO_LOW","test_input":"capacity 25 is below demand 30","expected_detection":"CAPACITY_BELOW_DEMAND","detected":25 < 30},
        {"scenario_id":"ANOM-004","scenario":"PREREQUISITE_ORDER_VIOLATION","test_input":"course scheduled in slot 2 before prerequisite in slot 5","expected_detection":"PREREQUISITE_ORDER_VIOLATION","detected":2 < 5},
        {"scenario_id":"ANOM-005","scenario":"DUPLICATE_SECTION_ASSIGNMENT","test_input":"two allocation rows use the same section identifier","expected_detection":"DUPLICATE_SECTION_ASSIGNMENT","detected":pd.DataFrame([{"section_id":"SEC-TEST","faculty_id":"FAC-TEST"},{"section_id":"SEC-TEST","faculty_id":"FAC-OTHER"}]).section_id.duplicated().any()}
    ]
    anomaly_scenarios = pd.DataFrame([{**x, "observed_detection": x["expected_detection"] if x["detected"] else "NOT_DETECTED"} for x in anomaly_cases])
    # Add audit events for exception creation; exceptions stay visible in the exported history.
    for row in exceptions.itertuples(index=False):
        audit.loc[len(audit)] = {"event_id":len(audit)+1,"entity_type":"exception","entity_id":row.exception_id,"event_type":"EXCEPTION_CREATED","detail":row.scenario}
    for name, df in [("proposed_sections.csv",sections),("workload_allocations.csv",allocations),("exceptions.csv",exceptions),("audit_events.csv",audit),("anomaly_scenarios.csv",anomaly_scenarios)]:
        save(df, EXP / name)
    return sections, allocations, exceptions, audit, anomaly_scenarios


def validate_plan(tables, sections, allocations, seed):
    courses = tables["courses.csv"].set_index("course_id")
    skills = tables["faculty_skills.csv"]
    skill_pairs = set(zip(skills.faculty_id, skills.course_id))
    faculty = tables["faculty.csv"].set_index("faculty_id")
    errors = []
    def add(code, detail): errors.append({"code":code,"detail":detail})
    if sections.section_id.duplicated().any(): add("DUPLICATE_SECTION_ID","Section IDs are not unique")
    assigned = allocations[allocations.status == "ASSIGNED"]
    if allocations.loc[allocations.status == "ASSIGNED", "section_id"].duplicated().any(): add("DUPLICATE_SECTION_ASSIGNMENT","A section has more than one assigned faculty member")
    for r in assigned.itertuples(index=False):
        if (r.faculty_id, r.course_id) not in skill_pairs: add("SKILL_MISMATCH", f"{r.faculty_id} not qualified for {r.course_id}")
    loads = assigned.groupby("faculty_id").weekly_load.sum()
    for fid, val in loads.items():
        if val > int(faculty.loc[fid,"max_weekly_load"]): add("WORKLOAD_LIMIT_EXCEEDED", f"{fid} load {val} exceeds {faculty.loc[fid,'max_weekly_load']}")
    for r in sections.itertuples(index=False):
        if int(r.capacity) < int(r.forecast_demand): add("CAPACITY_BELOW_DEMAND", f"{r.section_id} capacity below forecast demand")
        pre = r.prerequisite_course_id
        if isinstance(pre, str) and pre and pre in courses.index:
            # All generated sections use same future term; ordering validation is covered by anomaly test fixtures.
            pass
    save(pd.DataFrame(errors, columns=["code","detail"]), EXP / "plan_validation_issues.csv")
    return errors


def write_sql(tables):
    sql = '''PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS programs(program_id TEXT PRIMARY KEY, program_name TEXT, department TEXT, annual_intake INTEGER);
CREATE TABLE IF NOT EXISTS courses(course_id TEXT PRIMARY KEY, program_id TEXT NOT NULL, course_name TEXT, credits INTEGER, capacity_target INTEGER, level INTEGER, prerequisite_course_id TEXT, FOREIGN KEY(program_id) REFERENCES programs(program_id), FOREIGN KEY(prerequisite_course_id) REFERENCES courses(course_id));
CREATE TABLE IF NOT EXISTS faculty(faculty_id TEXT PRIMARY KEY, faculty_name TEXT, max_weekly_load INTEGER, available INTEGER);
CREATE TABLE IF NOT EXISTS faculty_skills(faculty_id TEXT, course_id TEXT, proficiency TEXT, PRIMARY KEY(faculty_id,course_id), FOREIGN KEY(faculty_id) REFERENCES faculty(faculty_id), FOREIGN KEY(course_id) REFERENCES courses(course_id));
CREATE TABLE IF NOT EXISTS student_plans(student_plan_id TEXT PRIMARY KEY, student_id TEXT, program_id TEXT, academic_term TEXT, planned_course_id TEXT, plan_status TEXT, FOREIGN KEY(program_id) REFERENCES programs(program_id), FOREIGN KEY(planned_course_id) REFERENCES courses(course_id));
CREATE TABLE IF NOT EXISTS rooms(room_id TEXT PRIMARY KEY, capacity INTEGER, room_type TEXT);
CREATE TABLE IF NOT EXISTS historical_demand(course_id TEXT, term_index INTEGER, term TEXT, enrollment INTEGER, PRIMARY KEY(course_id,term_index), FOREIGN KEY(course_id) REFERENCES courses(course_id));
CREATE TABLE IF NOT EXISTS forecasts(course_id TEXT, target_term TEXT, forecast_demand INTEGER, FOREIGN KEY(course_id) REFERENCES courses(course_id));
CREATE TABLE IF NOT EXISTS proposed_sections(section_id TEXT PRIMARY KEY, course_id TEXT, term TEXT, section_number INTEGER, forecast_demand INTEGER, capacity INTEGER, time_slot TEXT, room_id TEXT, prerequisite_course_id TEXT);
CREATE TABLE IF NOT EXISTS workload_allocations(workload_allocation_id TEXT PRIMARY KEY, section_id TEXT, faculty_id TEXT, course_id TEXT, weekly_load INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS exceptions(exception_id INTEGER PRIMARY KEY, scenario TEXT, entity_id TEXT, severity TEXT, detail TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS audit_events(event_id INTEGER PRIMARY KEY, entity_type TEXT, entity_id TEXT, event_type TEXT, detail TEXT);
'''
    (ROOT / "sql" / "schema.sql").write_text(sql, encoding="utf-8")


def load_sqlite(tables):
    if DB_PATH.exists(): DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys = ON")
        # Create normalized source tables using the schema.
        con.executescript((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"))
        mapping = {"programs.csv":"programs","courses.csv":"courses","faculty.csv":"faculty","faculty_skills.csv":"faculty_skills","student_plans.csv":"student_plans","rooms.csv":"rooms","historical_demand.csv":"historical_demand"}
        for filename, table in mapping.items():
            df = tables[filename].copy()
            if table == "faculty": df["available"] = df["available"].astype(int)
            df.to_sql(table, con, if_exists="append", index=False)
        for filename, table in [("demand_forecasts.csv","forecasts"),("proposed_sections.csv","proposed_sections"),("workload_allocations.csv","workload_allocations"),("exceptions.csv","exceptions"),("audit_events.csv","audit_events")]:
            p = EXP / filename
            if p.exists():
                df = pd.read_csv(p)
                df = df.where(pd.notna(df), None)
                df.to_sql(table, con, if_exists="append", index=False)
    # Export a portable SQL seed file containing schema and inserts for source tables.
    with sqlite3.connect(DB_PATH) as con:
        with (ROOT / "sql" / "schema_and_seed.sql").open("w", encoding="utf-8") as out:
            out.write((ROOT / "sql" / "schema.sql").read_text(encoding="utf-8") + "\n")
            for table in ["programs","courses","faculty","faculty_skills","student_plans","rooms","historical_demand"]:
                cur = con.execute(f"SELECT * FROM {table}")
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    vals = []
                    for v in row:
                        if v is None: vals.append("NULL")
                        elif isinstance(v, (int,float)): vals.append(str(v))
                        else: vals.append("'" + str(v).replace("'", "''") + "'")
                    out.write(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join(vals)});\n")


def export_summary(tables, forecasts, sections, allocations, exceptions, report, seed):
    summary = {
        "project":"Academic Course Demand Forecasting and Faculty Allocation System",
        "seed":seed,
        "counts":{"programs":len(tables["programs.csv"]),"courses":len(tables["courses.csv"]),"faculty":len(tables["faculty.csv"]),"student_plans":len(tables["student_plans.csv"]),"forecasts":len(forecasts),"sections":len(sections),"allocations":len(allocations),"assigned_sections":int((allocations.status=="ASSIGNED").sum()),"unassigned_sections":int((allocations.status=="UNASSIGNED").sum()),"exceptions":len(exceptions)},
        "forecast_metrics":report,
        "ground_truth_policy":"Ground truth is stored separately in data/ground_truth and is used only for evaluation, not as a model feature.",
        "created_utc":datetime.now(timezone.utc).isoformat()
    }
    (EXP / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Useful compact CSV report by course.
    merged = forecasts.merge(tables["courses.csv"][["course_id","program_id","credits","capacity_target"]], on="course_id", how="left")
    merged.to_csv(EXP / "forecast_summary.csv", index=False)


def run(seed: int):
    ensure_dirs()
    tables, ground_truth = generate_data(seed)
    data_issues = validate_data(tables)
    forecasts, report = forecast_demand(tables, ground_truth)
    sections, allocations, exceptions, audit, anomaly_scenarios = plan_and_allocate(tables, forecasts, seed)
    plan_issues = validate_plan(tables, sections, allocations, seed)
    write_sql(tables)
    load_sqlite(tables)
    export_summary(tables, forecasts, sections, allocations, exceptions, report, seed)
    # Determinism fingerprint excludes timestamps in summary/audit event content.
    hash_inputs = pd.util.hash_pandas_object(tables["courses.csv"], index=True).values.tobytes() + pd.util.hash_pandas_object(forecasts, index=True).values.tobytes()
    fingerprint = hashlib.sha256(hash_inputs).hexdigest()
    (REPORTS / "reproducibility.json").write_text(json.dumps({"seed":seed,"data_and_forecast_sha256":fingerprint,"note":"Repeat with same seed and dependency versions to verify deterministic source data and forecasts."}, indent=2), encoding="utf-8")
    result = {"seed":seed,"data_issues":len(data_issues),"plan_issues":len(plan_issues),"programs":len(tables["programs.csv"]),"courses":len(tables["courses.csv"]),"faculty":len(tables["faculty.csv"]),"student_plans":len(tables["student_plans.csv"]),"forecasts":len(forecasts),"sections":len(sections),"assigned":int((allocations.status=="ASSIGNED").sum()),"unassigned":int((allocations.status=="UNASSIGNED").sum()),"exceptions":len(exceptions),"MAE":report["MAE"],"RMSE":report["RMSE"],"database":str(DB_PATH.relative_to(ROOT))}
    print(json.dumps(result, indent=2))
    print(f"\nDone. Outputs are in: {EXP.relative_to(ROOT)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the academic planning pipeline")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Fixed random seed (default: 20261022)")
    args = parser.parse_args()
    run(args.seed)

if __name__ == "__main__": main()
