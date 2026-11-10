"""Verify the retained final experiment outputs without selecting models."""

from pathlib import Path
import os
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE / "src"), str(PACKAGE / "verification")]
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["DAG_OUTPUT_ROOT"] = str(PACKAGE / "outputs/runtime")

import argparse
import hashlib
from fractions import Fraction
import json
import numpy as np
import runtime346_v1 as runtime
from run_final376_v1 import read_plan


def power():
    import audit_power365_v1 as audit
    audit.ROOT = PACKAGE / "outputs/round365"
    audit.ASSETS = PACKAGE / "inputs"
    plan = read_plan(PACKAGE / "specifications/final_power_plan.json")["tasks"]
    tasks = []
    for source in plan:
        identifier = source["identifier"]
        tasks.append(dict(identifier="source_"+identifier, source=identifier, kind="source"))
        result = runtime.read(audit.ROOT / "conditions365_v1" / (identifier+".json"))["certificate"]
        if result["reported"]:
            for component, pair in enumerate(result["Fieller_components"]):
                lower, upper = map(Fraction, pair)
                for part in range(4):
                    tasks.append(dict(identifier="profile_%s_%d_%d" % (identifier, component, part), source=identifier,
                                      kind="profile", lower=str(lower+(upper-lower)*part/4),
                                      upper=str(lower+(upper-lower)*(part+1)/4)))
    report = runtime.execute("verify_final_power", audit.operate, tasks, maximum=128)
    fields = ("source_conditions", "covariance_entries", "raw_P_intervals", "Fieller_sets_proved", "false_report",
              "excluded_profile_parts", "rational_subregions", "Sturm_evaluations")
    summary = dict(status="passed" if report["status"] == "completed" else "errors_to_repair",
                   **{name: sum(row.get(name, 0) for row in report["records"]) for name in fields})
    runtime.save(PACKAGE / "outputs/verification/power.json", summary)
    print(json.dumps(summary), flush=True)


def connected():
    import audit_connected367_v3 as audit
    audit.ROOT = PACKAGE / "outputs/connected"
    tasks = read_plan(PACKAGE / "specifications/final_connected_plan.json")["tasks"]
    report = runtime.execute("verify_final_connected", audit.check, tasks, maximum=64)
    fields = ("symbolic_full_inverses", "connected_conditions", "omitted_target_witnesses",
              "complete_support_selections", "stable_feedback_models", "parent_domain_decisions")
    summary = dict(status="passed" if report["status"] == "completed" else "errors_to_repair",
                   **{name: sum(row.get(name, 0) for row in report["records"]) for name in fields})
    runtime.save(PACKAGE / "outputs/verification/connected.json", summary)
    print(json.dumps(summary), flush=True)


def joint():
    import audit_joint369_v2 as audit
    audit.ROOT = PACKAGE / "outputs/joint"
    audit.ASSETS = PACKAGE / "outputs"
    tasks = read_plan(PACKAGE / "specifications/final_power_plan.json")["tasks"]
    report = runtime.execute("verify_final_joint", audit.audit, tasks, maximum=128)
    fields = ("source_conditions", "unqualified", "coefficient_intervals", "rational_witnesses",
              "rational_point_inequalities", "excluded_joint_sets", "Sturm_evaluations", "false_reports", "unresolved")
    summary = dict(status="passed" if report["status"] == "completed" else "errors_to_repair",
                   **{name: sum(row.get(name, 0) for row in report["records"]) for name in fields})
    runtime.save(PACKAGE / "outputs/verification/joint.json", summary)
    print(json.dumps(summary), flush=True)


def optical_check(info):
    from scipy.stats import f
    rows = np.load(info["case"]["group_paths"][0])
    design = np.column_stack((np.ones(len(rows)), rows[:, :9]))
    response = rows[:, 9]
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > max(design.shape)*np.finfo(float).eps*singular[0]))
    coefficient = right[:rank].T @ ((left[:, :rank].T @ response) / singular[:rank])
    residual = response - design @ coefficient
    scale = float(residual @ residual) / (len(response)-rank)
    covariance = (right[:rank].T / singular[:rank]**2) @ right[:rank] * scale
    record = runtime.read(PACKAGE / "outputs/measured/optical" / ("experiment%d" % info["experiment"]) / "row_certificate/result.json")
    assert rank == record["rank"]
    checked = 0
    for index, row in enumerate(record["groups"]):
        if row["unresolved"]:
            continue
        statistic = coefficient[index+1]**2 / covariance[index+1, index+1]
        probability = float(f.sf(statistic, 1, len(rows)-rank))
        assert np.isclose(statistic, row["statistic"], rtol=1e-7, atol=1e-9)
        assert np.isclose(probability, row["original_P"], rtol=1e-7, atol=1e-12)
        assert row["rejected"] == (probability < .05/9.)
        checked += 1
    return dict(status="completed", experiment=info["experiment"], original_P_checks=checked,
                rank=rank, transformed_rows=len(rows))


def measured():
    plan = read_plan(PACKAGE / "specifications/final_measured_plan.json")
    checks = []
    for info in plan["cases"]:
        if info["panel"] == "optical":
            checks.append(optical_check(info))
    # Check the stored rank definition independently of the main combination code.
    from bisect import bisect_right
    from math import prod
    info = next(info for info in plan["cases"] if info["panel"] == "light")
    graphs, comparisons = [], 0
    for target in range(10):
        frames = []
        for environment in range(11):
            name = "e%d_t%d" % (environment, target)
            with np.load(PACKAGE / "outputs/measured/light/prepared" / name / "components.npz") as data:
                observed = data["observed"]
            blocks = [np.load(PACKAGE / "outputs/measured/light/reference_blocks" / name / ("b%05d.npy" % start))
                      for start in range(0, 32767, 512)]
            frames.append(np.concatenate((observed[None], *blocks)))
        for response in range(9):
            ranks = []
            for frame in frames:
                values = frame[:, response]
                ranks.append(32768-np.searchsorted(np.sort(values), values, side="left"))
            ranks = np.column_stack(ranks)
            legacy = 11*1800*ranks.min(axis=1) <= 32768
            products = [prod(int(x) for x in row) for row in ranks]
            first = sorted(value for value, active in zip(products, legacy) if active)
            second = sorted(value for value, active in zip(products, legacy) if not active)
            expected = np.array([bisect_right(first, value) if active else len(first)+bisect_right(second, value)
                                 for value, active in zip(products, legacy)])
            saved = np.load(PACKAGE / "outputs/measured/light/row_estimator/ranks" / ("t%d_h%d.npz" % (target,response)))["counts"]
            assert np.array_equal(saved, expected)
            if 1800*int(expected[0]) <= 32768:
                graphs.append([target, response+10])
            comparisons += len(expected)
    result = runtime.read(PACKAGE / "outputs/measured/light/row_estimator/native_record.json")
    assert result["qualified_edges"] == graphs
    # Reconstruct the condition-rank counts and the three-condition union.
    camera = runtime.read(PACKAGE / "outputs/measured/camera/row_estimator/native_record.json")
    probabilities = np.empty((3,5,10))
    for target in range(5):
        scores = np.concatenate([np.load(PACKAGE / "outputs/measured/camera/reference_blocks" /
                                 ("c%d_b%05d.npz" % (target,start)))["scores"] for start in range(0,32767,128)])
        observed = np.stack([np.load(PACKAGE / "outputs/measured/camera/prepared" /
                                   ("c%d_t%d.npz" % (condition,target)))["observed"] for condition in range(3)])
        probabilities[:,target] = (1+np.sum(scores >= observed[None],axis=0))/32768.
    combined = np.minimum(1.,3*probabilities.min(axis=0))
    assert np.array_equal(combined,camera["original_P"])
    edges = [[root,head+5] for root in range(5) for head in range(10) if combined[root,head] <= .001]
    assert edges == camera["qualified_edges"]
    report = dict(status="passed", optical=checks, light_rank_values=comparisons, camera_condition_P_values=150)
    runtime.save(PACKAGE / "outputs/verification/measured.json", report)
    print(json.dumps(report),flush=True)


def frozen_files():
    manifest = runtime.read(PACKAGE / "freeze.json")
    failures = []
    for row in manifest["files"]:
        path = PACKAGE / row["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append(row["path"])
    report = dict(status="passed" if not failures else "mismatch", checked_files=len(manifest["files"]),
                  failures=failures, freeze_id=manifest["freeze_id"], checked_beijing=runtime.stamp())
    runtime.save(PACKAGE / "outputs/verification/frozen_files.json", report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("power","connected","joint","measured","frozen_files"))
    args = parser.parse_args()
    globals()[args.stage]()
