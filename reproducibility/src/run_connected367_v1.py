"""Connected final population conditions."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import time
import runtime346_v1 as runtime
import response_factors356_v1 as factors
import partial_target_queries357_v1 as partial
from Gaussian_sources365_v1 import model, encoded
ROOT = Path(__file__).resolve().parent


def downstream(size, structure):
    matrix = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    if structure == "layered":
        for node in range(2, size):
            previous_layer = 2 * (node // 2 - 1)
            for parent in (previous_layer, previous_layer + 1):
                matrix[node][parent] = Fraction(1, 4)
    elif structure == "skip_chain":
        for node in range(1, size):
            matrix[node][node - 1] = Fraction(1, 4)
            if node >= 3:
                matrix[node][node - 3] = Fraction(1, 4)
    else:
        raise ValueError("Unknown declared downstream structure")
    inputs = [Fraction(node < 2, 64) for node in range(size)]
    response = factors.response_from_lower(matrix)
    propagation = [sum((value * weight for value, weight in zip(row, inputs)), Fraction(0))
                   for row in response]
    assert max(sum(abs(value) for value in row) for row in matrix) <= Fraction(1, 2)
    assert max(sum(abs(matrix[i][j]) for i in range(size)) for j in range(size)) <= Fraction(1, 2)
    assert sum(value * value for value in propagation) <= Fraction(1, 256)
    return matrix, inputs, response, propagation


def lift_columns(core_columns, response, propagation):
    size = len(response)
    columns = [tuple(column) + tuple(value * column[3] for value in propagation) for column in core_columns]
    columns.extend((Fraction(0),) * 4 + tuple(response[i][j] for i in range(size)) for j in range(size))
    return columns


def extend_coefficients(core, matrix, inputs):
    size = 4 + len(matrix)
    coefficients = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for i in range(4):
        coefficients[i][:4] = core[i]
    for i, row in enumerate(matrix):
        coefficients[i + 4][3] = inputs[i]
        coefficients[i + 4][4:] = row
    return coefficients


def verify_inverse(columns, coefficients):
    size = len(columns)
    for row, values in enumerate(coefficients):
        parents = [(parent, coefficient) for parent, coefficient in enumerate(values) if coefficient]
        for target, column in enumerate(columns):
            value = column[row] - sum((coefficient * column[parent] for parent, coefficient in parents), Fraction(0))
            assert value == int(row == target)


def edge_support(coefficients):
    return {(parent, child) for child, row in enumerate(coefficients) for parent, value in enumerate(row) if value}


def run_condition(task):
    gap = Fraction(task["gap"])
    source = model(task["index"], gap)
    matrix, inputs, response, propagation = downstream(task["nodes"] - 4, task["structure"])
    physical_columns = [tuple(source["response"][i][j] for i in range(4)) for j in range(4)]
    full_columns = lift_columns(physical_columns, response, propagation)
    candidates = [[tuple(column) + tuple(value * column[3] for value in propagation) for column in options]
                  for options in source["candidates"][:3]]
    candidates.extend([[column] for column in full_columns[3:]])
    physical = extend_coefficients(source["physical"], matrix, inputs)
    verify_inverse(full_columns, physical)
    scales = list(source["scales"])
    factor_columns = 0
    for target, options in enumerate(candidates):
        if target < 3:
            contrast = [[source["scales"][target] * value for value in row]
                        for row in factors.contrast(*options)]
        else:
            column = options[0]
            raw = [[a * b for b in column] for a in column]
            scale = Fraction(1, 4) / max(sum(abs(value) for value in row) for row in raw)
            scales.append(scale)
            contrast = [[scale * value for value in row] for row in raw]
        recovered = factors.factor_candidates(contrast, target)
        assert set(recovered) == set(options)
        factor_columns += len(recovered)

    admitted, cyclic, degree_rejected = [], 0, 0
    for choices in product((0, 1), repeat=3):
        columns = [candidates[j][choices[j]] for j in range(3)] + [options[0] for options in candidates[3:]]
        masks = [sum(1 << i for i, value in enumerate(column) if i != j and value)
                 for j, column in enumerate(columns)]
        if not factors.acyclic_masks(masks):
            cyclic += 1
            continue
        core = factors.coefficients_from_columns([column[:4] for column in columns[:4]])
        coefficients = extend_coefficients(core, matrix, inputs)
        verify_inverse(columns, coefficients)
        if sum(value != 0 for value in core[3]) > 2:
            degree_rejected += 1
            continue
        admitted.append(dict(choices=list(choices), shortcut=core[3][0], support=sorted(edge_support(coefficients))))
    assert (len(admitted), cyclic, degree_rejected) == (2, 4, 2)
    supports = {tuple(map(tuple, result["support"])) for result in admitted}
    assert len(supports) == (1 if gap else 2)
    assert tuple(sorted(edge_support(physical))) in supports

    comparisons = []
    for parent_count in (2, 4, 8):
        parents = (1, 2) + tuple(range(4, 4 + parent_count - 2))
        began = time.monotonic()
        result = partial.spanning_query(candidates, 3, parents, parents)
        assert result["feasible"] == (gap == 0)
        comparisons.append(dict(parent_count=parent_count, feasible=result["feasible"],
                                branches=result["branches"], seconds=time.monotonic() - began,
                                singular_fallback=result["spanning_singular_fallback"]))

    parameters = source["parameters"]
    a, c, b, k, e = (parameters[name] for name in ("a", "c", "b", "k", "e"))
    witnesses = []
    for omitted in range(3):
        core = [[Fraction(0) for _ in range(4)] for _ in range(4)]
        core[1][0] = a
        if omitted == 0:
            core[2][1], core[3][2] = c, b
        elif omitted == 1:
            core[2][0], core[3][2], core[3][1] = a * c, b, gap / a
        else:
            core[2][0], core[2][1], core[3][2] = a * (c - k), k, e
        coefficients = extend_coefficients(core, matrix, inputs)
        unknown_response = factors.response_from_lower(core)
        columns = lift_columns([tuple(unknown_response[i][j] for i in range(4)) for j in range(4)],
                               response, propagation)
        verify_inverse(columns, coefficients)
        assert core[3][0] == 0 and sum(value != 0 for value in core[3]) <= 2
        assert all(column in candidates[j] for j, column in enumerate(columns) if j != omitted)
        domains = [None if j == omitted else options for j, options in enumerate(candidates)]
        began = time.monotonic()
        query = partial.parent_query(domains, 3, (1, 2))
        assert query["feasible"]
        witnesses.append(dict(omitted=omitted, core=core, feasible=query["feasible"],
                              solver=query["solver"], seconds=time.monotonic() - began))

    record = dict(status="completed", **task, parameters=parameters, downstream=matrix,
                  downstream_inputs=inputs, propagation=propagation, candidates=candidates,
                  scales=scales, physical_support=sorted(edge_support(physical)), admitted=admitted,
                  factor_columns=factor_columns, response_cyclic=cyclic, degree_rejected=degree_rejected,
                  parent_domain_queries=comparisons, omitted_target_witnesses=witnesses,
                  covariance_eigenvalue_bounds=[Fraction(125, 1024), Fraction(1923, 1024)],
                  evaluation_scope="exact_population_connected_support_and_query_qualification")
    runtime.save(ROOT / "conditions367_v1" / (task["identifier"] + ".json"), encoded(record))
    return dict(status="completed", nodes=task["nodes"], structure=task["structure"], gap=task["gap"],
                factor_columns=factor_columns, admitted=len(admitted), unique_support=bool(gap),
                complete_target_queries=3, omitted_target_witnesses=3)
