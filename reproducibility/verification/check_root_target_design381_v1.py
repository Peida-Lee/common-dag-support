"""Exact checks of the root-target construction and the retained query solver."""

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone, timedelta


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = PROJECT / "round376/replay_build/src"
if not SOURCE.is_dir():
    SOURCE = PROJECT / "src"
sys.path.insert(0, str(SOURCE))
import response_factors356_v1 as common
import partial_target_queries357_v1 as partial


def multiply(first, second):
    return [[sum((a*b for a, b in zip(row, column)), Fraction(0))
             for column in zip(*second)] for row in first]


def check_case(case):
    size, mode, repeat = case
    rng = random.Random(3800000 + size*100 + mode*10 + repeat)
    roots = (0, 1)
    source, response = 0, size-1
    domains = [None] * size
    for target in roots:
        options = []
        for choice in range(2):
            column = [Fraction(i == target) for i in range(size)]
            for row in range(2, size):
                column[row] = Fraction(rng.choice((-2, -1, 0, 1, 2)), 3)
            if target == source:
                if mode == 1:
                    column[2:] = [Fraction(0)] * (size-2)
                    column[response] = Fraction(choice+1, 3)
                elif mode == 2 and choice == 0:
                    column[response] = Fraction(0)
                elif mode == 3:
                    column[2:] = [Fraction(0)] * (size-2)
                    column[response] = Fraction(1)
                    if choice == 1:
                        column[2] = Fraction(1, 2)
            options.append(tuple(column))
        domains[target] = list(dict.fromkeys(options))

    expected = any(column[response] == 0 or any(column[m] for m in range(2, response))
                   for column in domains[source])
    parents = tuple(i for i in range(size) if i not in (source, response))
    result = partial.parent_query(domains, response, parents, roots)
    assert result["feasible"] == expected
    untargeted = list(domains)
    untargeted[source] = None
    untargeted_result = partial.parent_query(untargeted, response, parents, roots)
    assert untargeted_result["feasible"]

    contrasts = {}
    for target in roots:
        options = domains[target]
        contrasts[target] = common.contrast(options[0], options[-1])
        recovered = common.factor_candidates(contrasts[target], target)
        assert set(recovered) == set(options)

    general_structural = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for row in range(2, size):
        for parent in range(2, row):
            general_structural[row][parent] = Fraction(rng.choice((-1, 0, 1)), 5)
        for target in roots:
            general_structural[row][target] = domains[target][0][row] - sum(
                (general_structural[row][parent]*domains[target][0][parent]
                 for parent in range(2, size)), Fraction(0))
    general_inverse = [[Fraction(i == j)-general_structural[i][j]
                        for j in range(size)] for i in range(size)]
    general_response = common.inverse(general_inverse)
    masks = [sum(1 << i for i in range(size) if general_structural[i][j])
             for j in range(size)]
    assert common.acyclic_masks(masks)
    for target in roots:
        assert tuple(row[target] for row in general_response) == domains[target][0]
        transformed = multiply(multiply(general_inverse, contrasts[target]),
                               list(map(list, zip(*general_inverse))))
        assert all(transformed[i][j] == 0 for i in range(size) for j in range(size)
                   if i != target and j != target)
    witnesses = 0
    for first in domains[source]:
        if first[response] and not any(first[m] for m in range(2, response)):
            continue
        columns = common.identity(size)
        # identity() is symmetric, so each row can initially serve as a column.
        columns[source] = list(first)
        columns[1] = list(domains[1][0])
        structural = common.coefficients_from_columns(columns)
        if first[response]:
            mediator = next(m for m in range(2, response) if first[m])
            coefficient = first[response] / first[mediator]
            structural[response][mediator] = coefficient
            for target in roots:
                structural[response][target] = (columns[target][response]
                                               - coefficient*columns[target][mediator])
        assert structural[response][source] == 0
        assert all(structural[i][i] == 0 for i in range(size))
        masks = [sum(1 << i for i in range(size) if structural[i][j]) for j in range(size)]
        assert common.acyclic_masks(masks)
        inverse_response = [[Fraction(i == j)-structural[i][j] for j in range(size)]
                            for i in range(size)]
        actual_response = common.inverse(inverse_response)
        for target in roots:
            assert tuple(row[target] for row in actual_response) == tuple(columns[target])
            transformed = multiply(multiply(inverse_response, contrasts[target]),
                                   list(map(list, zip(*inverse_response))))
            assert all(transformed[i][j] == 0 for i in range(size) for j in range(size)
                       if i != target and j != target)
        witnesses += 1
    return dict(size=size, mode=mode, repeat=repeat, missing_feasible=expected,
                retained_solver_agreement=True, exact_missing_witnesses=witnesses,
                solver=result["solver"], untargeted_missing_feasible=True,
                general_parameterization_checked=True, query_decisions=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "root_target_exact_checks381_v1.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    reserve_script = PROJECT / "round320/build_manuscript320_v2.py"
    if os.name == "nt" and reserve_script.is_file():
        spec = importlib.util.spec_from_file_location("reserve380", reserve_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        affinity = module.reserve_performance_cores()
    else:
        affinity = dict(workers=workers, inherited_cpu_affinity=True)
    cases = [(size, mode, repeat) for size in (4, 5, 6) for mode in range(4) for repeat in range(2)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check_case, cases))
    report = dict(status="passed", arithmetic="exact rational", cases=len(results),
                  workers=workers, queries=2*len(results), affinity=affinity, results=results,
                  created_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  scope="Algebraic validation of the root-target design proposition and retained solver")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
