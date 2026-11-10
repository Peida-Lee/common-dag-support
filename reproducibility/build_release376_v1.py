"""Package the reported final experiments and their frozen reproducibility inputs."""

from pathlib import Path
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
import zipfile
import sys
import numpy as np


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE / "src"))
from run_final376_v1 import read_plan, portable, runtime


def derived_inputs(plan):
    from run_expanded_real_baselines323_v1 import arrays
    files = set()
    checked_cases = set()
    checked_configs = 0
    for task in plan["tasks"]:
        if task["method"] != "WVM_author":
            continue
        config_path = Path(task["config"])
        config = read_plan(config_path)
        case,groups,panel,head = task["case"],task["groups"],task["panel"],task["head"]
        assert config["B"] == 200
        assert config["predictor_columns"] == [i+1 for i in case["root_indices"]]
        assert config["response_column"] == head+1
        if panel == "light":
            levels,seed = [1/1800.,.05],3489000+head
        elif panel == "wind":
            levels,seed = [.001388888888888889,.05],3233150+100*case["p"]+head
        elif panel == "camera":
            levels,seed = [.001,.05],3469000+head
        else:
            levels,seed = [.05,.05/9.],3589000+case["global_id"]
        assert config["alphas"] == levels and config["seed"] == seed
        files.add(config_path)
        csv = Path(config["csv"])
        files.add(csv)
        if csv not in checked_cases:
            values,_ = arrays(case,groups)
            if panel == "wind":
                transformed=[]
                for value in values:
                    k=np.arange(1,len(value),dtype=float)[:,None]
                    transformed.append(np.sqrt(k/(k+1))*(value[1:]-np.cumsum(value,axis=0)[:-1]/k))
                values=transformed
            expected=np.concatenate(values)
            actual=np.loadtxt(csv,delimiter=",")
            assert np.array_equal(expected,actual)
            assert config["counts"] == [len(value) for value in values]
            checked_cases.add(csv)
        checked_configs += 1
    inventory=[dict(path=path.relative_to(PACKAGE).as_posix(),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(files)]
    report=dict(status="passed",derivation="deterministic preprocessing from the frozen scientific code and inputs",
                checked_configurations=checked_configs,checked_standardized_input_cases=len(checked_cases),files=inventory)
    runtime.save(PACKAGE / "derived_input_manifest.json",report)
    return files


def execution_manifest(plan):
    freeze=runtime.read(PACKAGE / "freeze.json")
    frozen_time=datetime.fromisoformat(freeze["frozen_beijing"])
    entries=[]
    for stage in ("final_power","final_connected","final_joint","final_baselines_cpu","final_baselines_gpu"):
        folder=PACKAGE / "outputs/runtime/task_receipts" / stage
        for path in sorted(folder.glob("*.json")):
            record=runtime.read(path)
            assert record["status"] == "completed"
            finished=datetime.fromisoformat(record["beijing"])
            started=finished-timedelta(seconds=record["seconds"])
            assert started > frozen_time
            entries.append(dict(experiment=stage,identifier=record["identifier"],status=record["status"],
                                started_beijing=started.isoformat(),finished_beijing=finished.isoformat(),
                                seconds=record["seconds"]))
    counts={stage:sum(row["experiment"]==stage for row in entries) for stage in sorted({row["experiment"] for row in entries})}
    assert counts == dict(final_power=2048,final_connected=96,final_joint=2048,final_baselines_cpu=184,final_baselines_gpu=16)
    for stage in ("wind_queries","light_preparation","light_references","camera_references"):
        summary=runtime.read(PACKAGE / "outputs/runtime" / (stage+"_completion.json"))
        launch=runtime.read(PACKAGE / "outputs/runtime" / (stage+"_launch.json"))
        assert summary["status"] == "completed"
        assert datetime.fromisoformat(launch["beijing"]) > frozen_time
        entries.append(dict(experiment=stage,status=summary["status"],tasks=summary["expected"],
                            started_beijing=launch["beijing"],finished_beijing=summary["beijing"]))
    report=dict(status="passed",freeze_id=freeze["freeze_id"],frozen_beijing=freeze["frozen_beijing"],
                generated_beijing=runtime.stamp(),case_counts=counts,all_delivered_science_after_freeze=True,entries=entries)
    runtime.save(PACKAGE / "execution_manifest.json",report)


def main():
    results=runtime.read(PACKAGE / "outputs/final_tables/final_results.json")
    assert results["status"] == "completed"
    for name in ("power","connected","joint","measured","frozen_files"):
        assert runtime.read(PACKAGE / "outputs/verification" / (name+".json"))["status"] == "passed",name
    plan=read_plan(PACKAGE / "specifications/final_measured_plan.json")
    derived=derived_inputs(plan)
    execution_manifest(plan)
    freeze=runtime.read(PACKAGE / "freeze.json")
    paths={PACKAGE / row["path"] for row in freeze["files"]}
    paths.update(derived)
    paths.update(PACKAGE / name for name in ("freeze.json","execution_manifest.json","derived_input_manifest.json",
        "README.md","reproduce.sh","create_fresh_replay376_v1.py","verify_final376_v1.py","summarize_final376_v1.py",
        "plot_final376_v1.py","build_release376_v1.py","environment/python_requirements.txt"))
    paths.update((PACKAGE / "verification").glob("*.py"))
    for folder in ("outputs/round365","outputs/connected","outputs/joint","outputs/verification","outputs/final_tables"):
        paths.update(path for path in (PACKAGE / folder).rglob("*") if path.is_file() and path.suffix in (".json",".npz",".csv",".pdf",".png",".svg",".smt2"))
    for task in plan["tasks"]:
        record=Path(task["output"])
        assert record.exists() and runtime.read(record)["status"] == "completed"
        paths.add(record)
        if task["method"] == "WVM_author":
            paths.add(record.parent / "author_result.toml")
        if task["method"] == "MuDo_CoM":
            paths.add(record.parent / "fitted_parameters.npz")
    for panel in ("light","wind","camera"):
        paths.add(PACKAGE / "outputs/measured" / panel / "row_estimator/native_record.json")
    paths.update((PACKAGE / "outputs/measured/light/row_estimator/ranks").glob("*.npz"))
    paths.update((PACKAGE / "outputs/measured/wind/row_estimator/queries").glob("*.json"))
    for case in results["measured"]["panels"]["optical"]["cases"]:
        paths.add(PACKAGE / "outputs/measured/optical" / ("experiment%d" % case["experiment"]) / "row_estimator/native_record.json")
    frozen_paths={row["path"] for row in freeze["files"]}
    output=PACKAGE / "final_reproducibility376_v1.zip"
    index=[]
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(paths):
            relative=path.relative_to(PACKAGE).as_posix()
            content=path.read_bytes()
            if relative not in frozen_paths and path.suffix == ".json":
                content=(json.dumps(portable(json.loads(content)),indent=2,ensure_ascii=False)+"\n").encode()
            archive.writestr(relative,content)
            index.append(dict(path=relative,bytes=len(content)))
        archive.writestr("release_index.json",json.dumps(dict(final_only=True,files=index),indent=2)+"\n")
    receipt=dict(status="completed",archive=output.name,bytes=output.stat().st_size,
                 sha256=hashlib.sha256(output.read_bytes()).hexdigest(),files=len(index)+1,
                 freeze_id=freeze["freeze_id"],final_only=True)
    runtime.save(PACKAGE / "release_receipt376_v1.json",receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    main()
