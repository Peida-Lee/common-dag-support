"""Frozen final experiments, from declared inputs to scientific outputs."""

from pathlib import Path
import os
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "src"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.update(PYTHONNOUSERSITE="1", CASTLE_BACKEND="pytorch", WANDB_MODE="disabled")
os.environ["NUMBA_CACHE_DIR"] = str(PACKAGE / "environment/numba_cache")
os.environ["MPLCONFIGDIR"] = str(PACKAGE / "environment/matplotlib_cache")
os.environ["XDG_CACHE_HOME"] = str(PACKAGE / "environment/cache")
os.environ["DAG_OUTPUT_ROOT"] = str(PACKAGE / "outputs/runtime")

import argparse
import copy
import hashlib
import json
import subprocess
import time
import tomllib
import traceback
from fractions import Fraction

import numpy as np
import runtime346_v1 as runtime


METHODS = ("PC", "GES", "GIES", "Gaussian_BIC_root_domain", "DirectLiNGAM",
           "NOTEARS", "DAGMA", "GOLEM", "MuDo_CoM", "WVM_author")


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(PACKAGE), "${PACKAGE}")
    return value


def resolved(value):
    if isinstance(value, dict):
        return {key: resolved(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolved(item) for item in value]
    if isinstance(value, str):
        return value.replace("${PACKAGE}", str(PACKAGE))
    return value


def write_plan(path, value):
    runtime.save(path, portable(value))


def read_plan(path):
    return resolved(runtime.read(path))


def prepared_groups(panel, original_case, original_groups, raw_folder):
    import pandas as pd
    folder = PACKAGE / "inputs/prepared" / panel
    folder.mkdir(parents=True, exist_ok=True)
    case = {key: copy.deepcopy(original_case[key]) for key in
            ("global_id", "study", "dataset", "configuration", "template", "n", "p",
             "variables", "root_indices", "output_indices", "mandatory", "queries", "ancestors")}
    groups, pooled = [], []
    for index, original in enumerate(original_groups):
        source_name = original.get("source", original.get("environment", "") + ".csv")
        frame = pd.read_csv(raw_folder / source_name)
        names = original["names"]
        values = frame[names].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        if "rows" in original:
            assert len(values) == original["rows"]
        path = folder / (Path(source_name).stem + ".npz")
        extra = {"assignment": frame["flag"].to_numpy(dtype=int)} if "flag" in frame else {}
        np.savez_compressed(path, values=values, **extra)
        group = {key: original[key] for key in ("environment", "target", "outputs") if key in original}
        group.update(id=index, source=source_name, names=names, stored=str(path), rows=len(values))
        groups.append(group)
        positions = [names.index(name) for name in case["variables"]]
        pooled.append(values[:, positions])
    combined = np.concatenate(pooled)
    assert combined.shape == (case["n"], case["p"])
    assert np.all(np.std(combined, axis=0, ddof=1) > 0.)
    path = folder / "pooled_shared_rows.npy"
    np.save(path, combined)
    case["group_paths"] = [str(path)]
    return dict(panel=panel, case=case, groups=groups)


def wvm_config(case_info, head, folder):
    from run_expanded_real_baselines323_v1 import arrays
    panel, case, groups = case_info["panel"], case_info["case"], case_info["groups"]
    folder.mkdir(parents=True, exist_ok=True)
    shared = folder.parent / "shared_standardized_rows.csv"
    values, _ = arrays(case, groups)
    if panel == "wind":
        contrasts = []
        for value in values:
            k = np.arange(1, len(value), dtype=float)[:, None]
            contrasts.append(np.sqrt(k / (k + 1)) * (value[1:] - np.cumsum(value, axis=0)[:-1] / k))
        values = contrasts
    if not shared.exists():
        np.savetxt(shared, np.concatenate(values), delimiter=",", fmt="%.17g")
    counts = [len(value) for value in values]
    if panel == "light":
        levels, seed = [1. / 1800., .05], 3489000 + head
    elif panel == "camera":
        levels, seed = [.001, .05], 3469000 + head
    elif panel == "wind":
        levels, seed = [.001388888888888889, .05], 3233150 + 100 * case["p"] + head
    else:
        levels, seed = [.05, .05 / 9.], 3589000 + case["global_id"]
    config = dict(csv=str(shared), counts=counts, predictor_columns=[i+1 for i in case["root_indices"]],
                  response_column=head+1, alphas=levels, B=200, seed=seed,
                  input_role="within_group_orthogonal_contrasts" if panel == "wind" else "shared_grouped_rows")
    write_plan(folder / "config.json", config)
    return str(folder / "config.json")


def prepare():
    if (PACKAGE / "freeze.json").exists():
        raise RuntimeError("This package already has a frozen input specification")
    cases = []
    source = runtime.read(PACKAGE / "specifications/light_plan.json")
    cases.append(prepared_groups("light", source["case"], source["groups"],
                                PACKAGE / "inputs/lt_interventions_standard_v1"))
    design = runtime.read(PACKAGE / "specifications/wind_design.json")
    wind_case = runtime.read(PACKAGE / "specifications/wind_case.json")
    cases.append(prepared_groups("wind", wind_case, design["original_groups"],
                                PACKAGE / "inputs/wt_validate_v1"))
    camera = runtime.read(PACKAGE / "specifications/camera_plan.json")
    info = dict(panel="camera", case=camera["case"], groups=camera["groups"])
    pooled = []
    for group in info["groups"]:
        path = PACKAGE / "inputs/camera" / (group["condition"] + ".npz")
        group["stored"] = str(path)
        pooled.append(np.load(path)["values"])
    path = PACKAGE / "inputs/prepared/camera/pooled_shared_rows.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.concatenate(pooled))
    info["case"]["group_paths"] = [str(path)]
    cases.append(info)
    import optical_rows358_v1 as optical
    for experiment, path in optical.catalogue(PACKAGE / "inputs/optical_device").items():
        original = np.loadtxt(path)
        assert len(original) == 1000
        pixels, source, response = optical.parse_values(original, experiment)
        rows, mapping = optical.source_contrasts(np.column_stack((pixels / 255., response)), source)
        folder = PACKAGE / "inputs/prepared/optical" / ("experiment%d" % experiment)
        folder.mkdir(parents=True, exist_ok=True)
        pooled = folder / "pooled_transformed_rows.npy"
        np.save(pooled, rows)
        runtime.save(folder / "source_contrast_map.json", mapping)
        groups = []
        for block, selected in enumerate((rows[:len(rows)//2], rows[len(rows)//2:])):
            block_path = folder / ("block%d.npz" % block)
            np.savez(block_path, values=selected)
            groups.append(dict(stored=str(block_path), names=["pixel%d_degree1" % i for i in range(9)]+["Y"],
                               rows=len(selected), block=block))
        ancestry = np.zeros((10, 10), dtype=int)
        ancestry[9, :9] = -1
        np.fill_diagonal(ancestry, -1)
        case = dict(global_id=358000+2*experiment, dataset="Janzing_exp%d_linear" % experiment,
                    study="Janzing_optical", configuration="optical_device", template=experiment,
                    n=len(rows), p=10, variables=groups[0]["names"], group_paths=[str(pooled)],
                    root_indices=list(range(9)), output_indices=[9], mandatory=[],
                    queries=[dict(u=i, w=9) for i in range(9)], ancestors=ancestry.tolist())
        cases.append(dict(panel="optical", experiment=experiment, case=case, groups=groups))
    tasks = []
    for info in cases:
        name = info["panel"] if info["panel"] != "optical" else "optical/experiment%d" % info["experiment"]
        folder = PACKAGE / "outputs/measured" / name
        for method in METHODS:
            heads = info["case"]["output_indices"] if method == "WVM_author" else [None]
            for head in heads:
                destination = folder / method
                if head is not None:
                    destination = destination / ("response%d" % head)
                task = dict(identifier=name.replace("/", "_")+"__"+method+("_h%d" % head if head is not None else ""),
                            method=method, output=str(destination / "native_record.json"), **info)
                if head is not None:
                    task.update(head=head, config=wvm_config(info, head, destination))
                tasks.append(task)
        if info["panel"] == "optical":
            tasks.append(dict(identifier="optical_exp%d__row_certificate" % info["experiment"], method="row_certificate",
                              output=str(folder / "row_certificate/result.json"), **info))
    write_plan(PACKAGE / "specifications/final_measured_plan.json", dict(cases=cases, tasks=tasks, methods=METHODS))
    # Remove outcome-bearing historical records from the scientific design metadata.
    wind_tasks = [{key: value for key, value in row.items() if key != "native"} for row in design["tasks"]]
    write_plan(PACKAGE / "specifications/final_wind_queries.json", dict(tasks=wind_tasks, family_queries=308))
    print("FINAL_INPUTS_PREPARED", len(cases), "cases", len(tasks), "method tasks", flush=True)


def configure_power():
    import run_power365_v1 as power
    power.ROOT = PACKAGE / "outputs/round365"
    power.ASSETS = PACKAGE / "inputs"
    power.ROOT.mkdir(parents=True, exist_ok=True)
    (power.ROOT / "conditions365_v1").mkdir(exist_ok=True)
    return power


def power_tasks():
    power = configure_power()
    tasks = []
    for index in power.INDICES:
        for gap in power.GAPS:
            name = "template%d_gap%d_%d" % (index, gap.numerator, gap.denominator)
            runtime.save(power.ROOT / "models365_v1" / (name+".json"), power.encoded(power.model(index, gap)))
            for rows in power.BUDGETS:
                for replicate in range(32):
                    tasks.append(dict(identifier="%s_n%d_rep%02d" % (name, rows, replicate), kind="condition", model=name,
                                      index=index, gap_numerator=gap.numerator, gap_denominator=gap.denominator,
                                      rows=rows, replicate=replicate))
    return tasks


def connected_tasks():
    return [dict(identifier="%s_p%d_i%d_s%s" % (structure, nodes, index, gap.replace("/", "_")),
                 structure=structure, nodes=nodes, index=index, gap=gap)
            for structure in ("layered", "skip_chain") for nodes in (16, 32, 64)
            for index in (0, 1, 3, 4) for gap in ("0", "1/8", "1/4", "1/2")]


def freeze():
    if (PACKAGE / "freeze.json").exists():
        raise RuntimeError("A frozen package requires a separate version for any correction")
    write_plan(PACKAGE / "specifications/final_power_plan.json", dict(tasks=power_tasks(), conditions=2048))
    write_plan(PACKAGE / "specifications/final_connected_plan.json", dict(tasks=connected_tasks(), conditions=96))
    from importlib.metadata import distributions
    installed = {distribution.metadata["Name"]: distribution.version for distribution in distributions()}
    runtime.save(PACKAGE / "environment/installed_versions.json", installed)
    paths = []
    for folder in ("inputs", "src", "specifications", "vendor"):
        paths.extend(path for path in (PACKAGE / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".nbc", ".nbi"))
    paths.extend((PACKAGE / "outputs/round365/models365_v1").glob("*.json"))
    paths.extend([Path(__file__), PACKAGE / "preflight_final376_v1.py", PACKAGE / "environment/installed_versions.json",
                  PACKAGE / "environment/WVM/Project.toml", PACKAGE / "environment/WVM/Manifest.toml"])
    inventory = [dict(path=path.relative_to(PACKAGE).as_posix(), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(set(paths))]
    plan = dict(version="final-replay-376-v1", frozen_beijing=runtime.stamp(), files=inventory,
                simulated_conditions=2048, connected_population_conditions=96,
                measured_configurations=["light", "wind", "camera", "optical"],
                model_selection_during_replay=False, real_data_role="fixed_final_method_evaluation_on_reused_measurements",
                seeds=dict(power=[365001, "index", "gap_numerator", "gap_denominator", "rows", "replicate", "cell"],
                           light_permutations=[347, 32767, "rows", "draw"], camera_trajectories=[346, 32767, "control", "draw"]),
                final_optical_basis="linear", final_optical_inheritance="MuDo_CoM_union_group_F", final_only=True)
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["freeze_id"] = hashlib.sha256(payload).hexdigest()
    runtime.save(PACKAGE / "freeze.json", plan)
    print("FROZEN", plan["freeze_id"], plan["frozen_beijing"], flush=True)


def require_freeze():
    info = runtime.read(PACKAGE / "freeze.json")
    runtime.ROOT = PACKAGE / "outputs/runtime"
    runtime.ROOT.mkdir(parents=True, exist_ok=True)
    return info


def run_power():
    require_freeze()
    power = configure_power()
    tasks = read_plan(PACKAGE / "specifications/final_power_plan.json")["tasks"]
    result = runtime.execute("final_power", power.condition, tasks, maximum=192)
    runtime.save(PACKAGE / "outputs/power_summary.json", result)


def run_connected():
    require_freeze()
    import run_connected367_v1 as connected
    connected.ROOT = PACKAGE / "outputs/connected"
    tasks = read_plan(PACKAGE / "specifications/final_connected_plan.json")["tasks"]
    result = runtime.execute("final_connected", connected.run_condition, tasks, maximum=96)
    runtime.save(PACKAGE / "outputs/connected_summary.json", result)


def run_joint():
    require_freeze()
    import joint_reference369_v2 as joint
    joint.ASSETS = PACKAGE / "outputs"
    joint.ROOT = PACKAGE / "outputs/joint"
    tasks = [dict(identifier=row["identifier"], source=row["identifier"])
             for row in read_plan(PACKAGE / "specifications/final_power_plan.json")["tasks"]]
    result = runtime.execute("final_joint", joint.condition, tasks, maximum=128)
    runtime.save(PACKAGE / "outputs/joint_summary.json", result)


def light_preparation(task):
    from public_covariate333_v1 import polynomial_adjust
    from joint_moment347_v1 import response_features
    from joint_randomization349_v1 import canonical_features, reference_frame
    values = np.load(task["group"]["stored"])["values"]
    target = task["target"]
    controls = values[:, [i for i in range(10) if i != target]]
    raw = values[:, 10:]
    adjusted, model = polynomial_adjust(controls, raw)
    adjusted[:, np.all(raw == raw[0], axis=0)] = 0.
    feature, normalizer = canonical_features(values[:, target], task["group"]["names"][target])
    responses = np.ascontiguousarray(response_features(raw, adjusted)[:, 18:])
    folder = PACKAGE / "outputs/measured/light/prepared" / task["identifier"]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "components.npz", features=feature, responses=responses,
             observed=reference_frame(feature, responses, np.empty((0, len(values)), dtype=np.uint16))[0, 1])
    runtime.save(folder / "nuisance.json", dict(model=model, target_normalization=normalizer))
    return dict(status="completed", rows=len(values))


def light_reference(task):
    from joint_randomization349_v1 import reference_frame
    path = PACKAGE / "outputs/measured/light/prepared" / ("e%d_t%d" % (task["environment"], task["target"]))
    with np.load(path / "components.npz") as data:
        feature, responses = data["features"], data["responses"]
    n = feature.shape[1]
    permutations = np.empty((task["end"]-task["start"], n), dtype=np.uint16)
    for row, index in enumerate(range(task["start"], task["end"])):
        permutations[row] = np.random.default_rng(np.random.SeedSequence([347, 32767, n, index])).permutation(n)
    scores = reference_frame(feature, responses, permutations)[:, 1]
    folder = PACKAGE / "outputs/measured/light/reference_blocks" / ("e%d_t%d" % (task["environment"], task["target"]))
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / ("b%05d.npy" % task["start"]), scores[1:])
    return dict(status="completed", start=task["start"], end=task["end"])


def run_light():
    require_freeze()
    plan = read_plan(PACKAGE / "specifications/final_measured_plan.json")
    source = next(info for info in plan["cases"] if info["panel"] == "light")
    tasks = [dict(identifier="e%d_t%d" % (environment, target), environment=environment, target=target, group=group)
             for environment, group in enumerate(source["groups"]) for target in range(10)]
    preparation = runtime.execute("light_preparation", light_preparation, tasks, maximum=64)
    if preparation["status"] != "completed":
        return
    references = [dict(identifier="e%d_t%d_b%05d" % (row["environment"], row["target"], start),
                       environment=row["environment"], target=row["target"], start=start, end=min(start+512, 32767))
                  for row in tasks for start in range(0, 32767, 512)]
    result = runtime.execute("light_references", light_reference, references, maximum=128)
    if result["status"] != "completed":
        return
    from joint_randomization349_v1 import combine_ranks
    records, graph = [], []
    frames = {}
    for row in tasks:
        name = row["identifier"]
        with np.load(PACKAGE / "outputs/measured/light/prepared" / name / "components.npz") as data:
            observed = data["observed"]
        blocks = [np.load(PACKAGE / "outputs/measured/light/reference_blocks" / name / ("b%05d.npy" % start))
                  for start in range(0, 32767, 512)]
        frames[row["environment"], row["target"]] = np.concatenate((observed[None], *blocks))
    for target in range(10):
        for response in range(9):
            scores = np.column_stack([frames[environment, target][:, response] for environment in range(11)])
            counts, metadata = combine_ranks(scores, list(range(11)))
            probability = Fraction(int(counts[0, 2]), 32768)
            row = dict(target=target, response=response, original_P=str(probability), original_P_display=float(probability),
                       rejected=probability <= Fraction(1, 1800), metadata=metadata)
            records.append(row)
            if row["rejected"]:
                graph.append([target, response+10])
            path = PACKAGE / "outputs/measured/light/row_estimator/ranks" / ("t%d_h%d.npz" % (target, response))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, counts=counts[:, 2].astype(np.uint16))
    runtime.save(PACKAGE / "outputs/measured/light/row_estimator/native_record.json",
                 dict(status="completed", method="legacy_joint_all_retained_Fisher", qualified_edges=graph,
                      records=records, original_rows=20000, B=32767, truth_read=False))


def camera_reference(task):
    from camera_CRT346_v1 import draw_prefixes, commands, MEANS, statistics
    values = draw_prefixes(task["target"], range(task["start"], task["end"]), 30000, 4096, 32767)
    scores = []
    for condition in range(3):
        prepared = np.load(PACKAGE / "outputs/measured/camera/prepared" / ("c%d_t%d.npz" % (condition, task["target"])))
        command = commands(values, task["target"], MEANS[condition][task["target"]])
        scores.append(statistics(command, task["target"], prepared["raw"], prepared["adjusted"])[:, 3])
    path = PACKAGE / "outputs/measured/camera/reference_blocks" / (task["identifier"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=np.stack(scores, axis=1))
    return dict(status="completed", target=task["target"], start=task["start"], end=task["end"])


def run_camera():
    require_freeze()
    from camera_CRT346_v1 import residuals, statistics, inclusive_pvalues
    info = next(info for info in read_plan(PACKAGE / "specifications/final_measured_plan.json")["cases"]
                if info["panel"] == "camera")
    observed = np.empty((3, 5, 10))
    for condition, group in enumerate(info["groups"]):
        values = np.load(group["stored"])["values"]
        for target in range(5):
            raw, adjusted, projection = residuals(values[:, :5], values[:, 5:], target, condition)
            observed[condition, target] = statistics(values[:, target], target, raw, adjusted)[0, 3]
            path = PACKAGE / "outputs/measured/camera/prepared" / ("c%d_t%d.npz" % (condition, target))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, raw=raw, adjusted=adjusted, observed=observed[condition, target])
            runtime.save(path.with_suffix(".json"), projection)
    tasks = [dict(identifier="c%d_b%05d" % (target, start), target=target, start=start, end=min(start+128, 32767))
             for target in range(5) for start in range(0, 32767, 128)]
    result = runtime.execute("camera_references", camera_reference, tasks, maximum=128)
    if result["status"] != "completed":
        return
    probabilities = np.empty((3, 5, 10))
    reports = []
    for target in range(5):
        scores = np.concatenate([np.load(PACKAGE / "outputs/measured/camera/reference_blocks" /
                                         (task["identifier"] + ".npz"))["scores"] for task in tasks if task["target"] == target])
        probabilities[:, target], counts = inclusive_pvalues(observed[:, target], scores)
        reports.append(dict(target=target, counts=counts.tolist(), original_P=probabilities[:, target].tolist()))
    combined = np.minimum(1., 3 * probabilities.min(axis=0))
    graph = [[root, 5+head] for root in range(5) for head in range(10) if combined[root, head] <= .001]
    runtime.save(PACKAGE / "outputs/measured/camera/row_estimator/native_record.json",
                 dict(status="completed", method="adjusted_expanded", qualified_edges=graph, original_rows=12288,
                      original_P=combined.tolist(), reports=reports, B=32767, truth_read=False))


def wind_query(task):
    import pandas as pd
    from ordered_projection345_v2 import PRESSURES, evaluate
    values = pd.read_csv(PACKAGE / "inputs/wt_validate_v1" / task["source"])
    assert len(values) == task["n"]
    result = evaluate(values["flag"].to_numpy(dtype=int), values[task["response"]].to_numpy(dtype=float),
                      308, adjust=task["response"] in PRESSURES)
    selected = result["methods"]["Selected_KS"]
    row = dict(status="completed", **task, selected=selected,
               original_P=selected["raw_P_exact"], rejected=Fraction(selected["raw_P_exact"]) <= Fraction(1, 6160))
    runtime.save(PACKAGE / "outputs/measured/wind/row_estimator/queries" / (task["identifier"]+".json"), row)
    return row


def run_wind():
    require_freeze()
    tasks = read_plan(PACKAGE / "specifications/final_wind_queries.json")["tasks"]
    result = runtime.execute("wind_queries", wind_query, tasks, maximum=64)
    if result["status"] != "completed":
        return
    info = next(info for info in read_plan(PACKAGE / "specifications/final_measured_plan.json")["cases"] if info["panel"] == "wind")
    names = info["case"]["variables"]
    graph = sorted({(names.index(row["target"]), names.index(row["response"])) for row in result["records"] if row["rejected"]})
    runtime.save(PACKAGE / "outputs/measured/wind/row_estimator/native_record.json",
                 dict(status="completed", method="Selected_KS", qualified_edges=graph, original_rows=23449,
                      query_count=308, truth_read=False))


def fit_wvm(task):
    config = read_plan(task["config"])
    destination = Path(task["output"]).parent
    runtime_config = destination / "config.toml"
    lines = []
    for key in ("csv", "counts", "predictor_columns", "response_column", "alphas", "B", "seed"):
        lines.append(key + " = " + json.dumps(config[key]))
    runtime_config.write_text("\n".join(lines)+"\n")
    native = destination / "author_result.toml"
    env = dict(os.environ, JULIA_DEPOT_PATH=os.environ.get("DAG_JULIA_DEPOT", str(PACKAGE / "environment/julia_depot")),
               JULIA_PKG_PRECOMPILE_AUTO="0", LD_LIBRARY_PATH="", LD_PRELOAD="")
    julia = os.environ.get("DAG_JULIA", str(PACKAGE / "environment/julia/bin/julia"))
    command = [julia, "--project="+str(PACKAGE / "environment/WVM"), "--startup-file=no",
               str(PACKAGE / "src/native_WVM332_v1.jl"), str(runtime_config), str(native),
               str(PACKAGE / "vendor/WVM/WassersteinVariance.jl")]
    with (destination / "author.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        code = process.wait()
    if code:
        raise RuntimeError("WVM author process returned %d" % code)
    author = tomllib.loads(native.read_text())
    from wvm_output_adapter335_v1 import fit_wvm_output
    record = fit_wvm_output(dict(task, config=str(runtime_config)), lambda task: None)
    record["metadata"].update(optimization_replayed=True, calibration="author_bootstrap_moments_Gamma", source_commit="ff98a88c81020d6dee74462625696ef17cedcbd5")
    return record


def baseline_task(task):
    from baseline_worker321_v2 import qualified
    output = Path(task["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    method = task["method"]
    if method == "row_certificate":
        from optical_rows358_v1 import group_tests
        rows = np.load(task["case"]["group_paths"][0])
        record = group_tests(rows[:, :9], rows[:, 9], [[i] for i in range(9)])
    elif method in ("PC", "GES", "DirectLiNGAM", "DAGMA", "NOTEARS", "Gaussian_BIC_root_domain"):
        from evaluate_randomized_inputs323_v3 import fit_baseline
        fit_baseline(task)
        record = runtime.read(output)
    elif method == "GIES":
        from run_baselines346_v1 import hardware_gies
        record = hardware_gies(task)
    elif method == "GOLEM":
        import golem_worker321_v1 as golem
        golem.BASE = PACKAGE / "specifications/verification"
        record = golem.fit(task)
    elif method == "MuDo_CoM":
        from run_expanded_real_baselines323_v1 import fit_mudo
        record = fit_mudo(task)
    else:
        record = fit_wvm(task)
    if record["status"] == "completed" and method != "row_certificate":
        native = set(map(tuple, record.get("native_edges", []))) | set(map(tuple, record.get("native_undirected_edges", [])))
        record["qualified_edges"] = sorted(map(list, qualified(native, task["case"])))
        if task["panel"] == "optical":
            from optical_rows358_v1 import pixel_projection
            record["detected_pixels"] = pixel_projection(record["qualified_edges"], [[i] for i in range(9)], 9)
    record.update(method=method, panel=task["panel"], experiment=task.get("experiment"),
                  started_epoch=started, finished_epoch=time.time(), truth_read=False,
                  freeze_id=runtime.read(PACKAGE / "freeze.json")["freeze_id"])
    runtime.save(output, record)
    return dict(status=record["status"], method=method, output=str(output), seconds=time.time()-started,
                traceback=record.get("traceback"))


def run_baselines(gpu):
    require_freeze()
    tasks = read_plan(PACKAGE / "specifications/final_measured_plan.json")["tasks"]
    tasks = [task for task in tasks if (task["method"] == "MuDo_CoM") == gpu]
    priority = {"MuDo_CoM": 0, "GOLEM": 0, "DAGMA": 1, "WVM_author": 2}
    tasks.sort(key=lambda task: (priority.get(task["method"], 3), task["identifier"]))
    result = runtime.execute("final_baselines_gpu" if gpu else "final_baselines_cpu", baseline_task, tasks,
                             maximum=4 if gpu else 64, start_method="spawn" if gpu else "fork")
    runtime.save(PACKAGE / "outputs" / ("baseline_gpu_summary.json" if gpu else "baseline_cpu_summary.json"), result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "freeze", "power", "connected", "joint", "light", "camera", "wind", "baseline_cpu", "baseline_gpu"))
    args = parser.parse_args()
    operations = dict(prepare=prepare, freeze=freeze, power=run_power, connected=run_connected,
                      joint=run_joint, light=run_light, camera=run_camera, wind=run_wind,
                      baseline_cpu=lambda: run_baselines(False), baseline_gpu=lambda: run_baselines(True))
    operations[args.stage]()


if __name__ == "__main__":
    main()
