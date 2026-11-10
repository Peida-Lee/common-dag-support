"""Generate final paper tables from every declared experiment output."""

from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import csv
import json
import sys
import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE / "src")]
from run_final376_v1 import METHODS, read_plan, runtime


def score(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    tp, fp, fn = len(predicted & reference), len(predicted-reference), len(reference-predicted)
    denominator = 2*tp+fp+fn
    value = Fraction(2*tp,denominator) if denominator else Fraction(1)
    return dict(TP=tp,FP=fp,FN=fn,SHD=fp+fn,F1=float(value),F1_exact=str(value))


def completion_bounds(candidate, baseline, positives, domain):
    unknown = domain-positives
    regions = ((candidate-baseline)&unknown,(baseline-candidate)&unknown,candidate&baseline&unknown,
               unknown-(candidate|baseline))
    hits1, hits2 = len(candidate&positives),len(baseline&positives)
    differences = []
    for a,b,c,d in product(*(range(len(region)+1) for region in regions)):
        total = len(positives)+a+b+c+d
        differences.append(Fraction(2*(hits1+a+c),len(candidate)+total)-
                           Fraction(2*(hits2+b+c),len(baseline)+total))
    return [str(min(differences)),str(max(differences))]


def simulation_summary():
    plan = read_plan(PACKAGE / "specifications/final_power_plan.json")["tasks"]
    settings = defaultdict(lambda: Counter(cases=0,qualified=0,reports=0,original=0,ordered=0,one_box=0))
    reasons, joint = Counter(),Counter()
    failures = []
    for task in plan:
        path = PACKAGE / "outputs/round365/conditions365_v1" / (task["identifier"]+".json")
        if not path.exists():
            failures.append(str(path))
            continue
        record = runtime.read(path)
        certificate = record["certificate"]
        key = (task["index"],str(Fraction(task["gap_numerator"],task["gap_denominator"])),task["rows"])
        cell = settings[key]
        cell.update(cases=1,qualified=int(certificate["ancestry"]["qualified"]),reports=int(certificate["reported"]),
                    original=int(record["original_interval"]["reported"]),ordered=int(record["order_qualified_interval"]["reported"]),
                    one_box=int(record["one_box_reported"]))
        reasons[certificate["reason"]] += 1
        joint_path = PACKAGE / "outputs/joint/conditions369_v2" / (task["identifier"]+".json")
        if not joint_path.exists():
            failures.append(str(joint_path))
            continue
        reference = runtime.read(joint_path)
        joint["reports"] += int(reference["reported"])
        joint["disagreements"] += int(reference["reported"] != certificate["reported"])
        joint["decision_"+reference["decision"]] += 1
        joint["certification_"+reference["certification"]] += 1
    rows = [dict(template=k[0],gap=k[1],rows=k[2],**v) for k,v in sorted(settings.items())]
    null = [row for row in rows if row["gap"] == "0"]
    return dict(status="completed" if not failures else "incomplete",settings=rows,reasons=dict(reasons),joint=dict(joint),
                conditions=sum(row["cases"] for row in rows),reports=sum(row["reports"] for row in rows),
                qualified=sum(row["qualified"] for row in rows),null_reports=sum(row["reports"] for row in null),
                missing=failures)


def measured_summary():
    labels = runtime.read(PACKAGE / "specifications/reference_labels.json")
    plan = read_plan(PACKAGE / "specifications/final_measured_plan.json")
    results, optical, missing = {},[],[]
    for info in plan["cases"]:
        panel = info["panel"]
        name = panel if panel != "optical" else "optical/experiment%d" % info["experiment"]
        folder = PACKAGE / "outputs/measured" / name
        graphs = {}
        for method in METHODS:
            tasks = [task for task in plan["tasks"] if task["case"]["global_id"] == info["case"]["global_id"]
                     and task["panel"] == panel and task["method"] == method]
            records = []
            for task in tasks:
                path = Path(task["output"])
                if not path.exists() or runtime.read(path).get("status") != "completed":
                    missing.append(task["identifier"])
                else:
                    records.append(runtime.read(path))
            if len(records) == len(tasks):
                graphs[method] = {tuple(edge) for record in records for edge in record["qualified_edges"]}
        if panel == "optical":
            path = folder / "row_certificate/result.json"
            if not path.exists() or "MuDo_CoM" not in graphs:
                missing.append(name+"__final_row_estimator")
                continue
            certificates = set(runtime.read(path)["detected_pixels"])
            inherited = {parent for parent,child in graphs["MuDo_CoM"] if child == 9}
            selected = inherited|certificates
            truth = set(labels["optical"][str(info["experiment"])] )
            scores = {method:score({parent for parent,child in graph if child==9},truth)
                      for method,graph in graphs.items()}
            scores["row_estimator"] = score(selected,truth)
            result = dict(experiment=info["experiment"],scores=scores,reference_pixels=sorted(truth),
                          predicted_pixels=sorted(selected),inherited_pixels=sorted(inherited),
                          certified_pixels=sorted(certificates),added_true=len((selected-inherited)&truth),
                          added_false=len((selected-inherited)-truth))
            optical.append(result)
            runtime.save(folder / "row_estimator/native_record.json",dict(status="completed",**result))
        else:
            path = folder / "row_estimator/native_record.json"
            if not path.exists():
                missing.append(name+"__row_estimator")
                continue
            graphs["row_estimator"] = set(map(tuple,runtime.read(path)["qualified_edges"]))
            variables = info["case"]["variables"]
            named = {method:{(variables[a],variables[b]) for a,b in graph} for method,graph in graphs.items()}
            truth = set(map(tuple,labels[panel]))
            scores = {method:dict(score(graph,truth),edges=sorted(map(list,graph))) for method,graph in named.items()}
            external = [method for method in METHODS if method in scores]
            strongest = max(external,key=lambda method:scores[method]["F1"])
            results[panel] = dict(scores=scores,strongest_baseline=strongest,
                                  gain_pp=100*(scores["row_estimator"]["F1"]-scores[strongest]["F1"]))
            if panel == "camera":
                domain = {(variables[a],variables[b]) for a in info["case"]["root_indices"] for b in info["case"]["output_indices"]}
                results[panel]["best_baseline_completion_difference"] = completion_bounds(named["row_estimator"],named[strongest],truth,domain)
    if len(optical) == 13 and all(len(case["scores"]) == 11 for case in optical):
        aggregate = {}
        for method in (*METHODS,"row_estimator"):
            counts = {field:sum(case["scores"][method][field] for case in optical) for field in ("TP","FP","FN","SHD")}
            denominator = 2*counts["TP"]+counts["FP"]+counts["FN"]
            value = Fraction(2*counts["TP"],denominator) if denominator else Fraction(1)
            aggregate[method] = dict(**counts,F1=float(value),F1_exact=str(value),
                                     macro_F1=float(np.mean([case["scores"][method]["F1"] for case in optical])),
                                     perfect_cases=sum(case["scores"][method]["SHD"]==0 for case in optical))
        strongest = max(METHODS,key=lambda method:aggregate[method]["F1"])
        results["optical"] = dict(scores=aggregate,strongest_baseline=strongest,
                                   gain_pp=100*(aggregate["row_estimator"]["F1"]-aggregate[strongest]["F1"]),cases=optical,
                                   added_true=sum(case["added_true"] for case in optical),
                                   added_false=sum(case["added_false"] for case in optical))
    return dict(status="completed" if not missing and len(results)==4 else "incomplete",panels=results,missing=missing)


def main():
    output = PACKAGE / "outputs/final_tables"
    output.mkdir(parents=True,exist_ok=True)
    simulation = simulation_summary()
    measured = measured_summary()
    report = dict(status="completed" if simulation["status"]==measured["status"]=="completed" else "incomplete",
                  freeze_id=runtime.read(PACKAGE / "freeze.json")["freeze_id"],simulation=simulation,measured=measured)
    runtime.save(output / "final_results.json",report)
    if simulation["settings"]:
        with (output / "power_settings.csv").open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(simulation["settings"][0]));writer.writeheader();writer.writerows(simulation["settings"])
    print(json.dumps(dict(status=report["status"],simulation_conditions=simulation["conditions"],reports=simulation["reports"],
                         null_reports=simulation["null_reports"],joint=simulation["joint"],
                         measured={panel:dict(F1=info["scores"]["row_estimator"]["F1"],best=info["strongest_baseline"],gain_pp=info["gain_pp"])
                                   for panel,info in measured["panels"].items()},missing=len(measured["missing"])),indent=2),flush=True)
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
