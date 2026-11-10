"""Evaluate the deterministic all-slot reference on the paper's fixed domains."""

from fractions import Fraction
from pathlib import Path
import argparse
import json
import zipfile


def scores(predicted, reference):
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn)
    return dict(TP=tp, FP=fp, FN=fn, SHD=fp + fn,
                precision=float(Fraction(tp, tp + fp)),
                recall=float(Fraction(tp, tp + fn)) if tp + fn else 0.,
                F1=float(f1), F1_exact=str(f1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = ("specifications/reference_labels.json",
             "specifications/final_measured_plan.json",
             "outputs/final_tables/final_results.json")
    if args.package.is_file():
        with zipfile.ZipFile(args.package) as archive:
            labels, plan, results = [json.loads(archive.read(name)) for name in names]
    else:
        labels, plan, results = [json.loads((args.package / name).read_text()) for name in names]
    panels = {}
    for panel in ("light", "wind", "camera"):
        case = next(item["case"] for item in plan["cases"] if item["panel"] == panel)
        domain = {(case["variables"][parent], case["variables"][child])
                  for parent in case["root_indices"] for child in case["output_indices"]}
        truth = set(map(tuple, labels[panel]))
        assert truth <= domain
        panels[panel] = scores(domain, truth)
    cases = []
    for name, reference in sorted(labels["optical"].items(), key=lambda item: int(item[0])):
        cases.append(dict(experiment=int(name), **scores(set(range(9)), set(reference))))
    tp = sum(case["TP"] for case in cases)
    fp = sum(case["FP"] for case in cases)
    fn = sum(case["FN"] for case in cases)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn)
    optical = results["measured"]["panels"]["optical"]["scores"]["row_estimator"]
    panels["optical"] = dict(TP=tp, FP=fp, FN=fn, SHD=fp + fn,
                            precision=float(Fraction(tp, tp + fp)), recall=1.,
                            F1=float(f1), F1_exact=str(f1),
                            macro_F1=sum(case["F1"] for case in cases) / len(cases),
                            perfect_cases=sum(case["SHD"] == 0 for case in cases),
                            cases=cases)
    current = dict(optical, precision=float(Fraction(optical["TP"], optical["TP"] + optical["FP"])),
                   recall=float(Fraction(optical["TP"], optical["TP"] + optical["FN"])))
    report = dict(status="completed", prediction_rule="report every allowed relation slot",
                  uses_measurement_values=False, panels=panels, optical_row_estimator=current)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
