from pathlib import Path
from itertools import product
import sympy as sp
import runtime346_v1 as runtime
ROOT = Path(__file__).resolve().parent.parent / 'outputs/connected' 


def rational_matrix(rows):
    return sp.Matrix([[sp.Rational(str(value)) for value in row] for row in rows])


def acyclic_columns(matrix):
    remaining = set(range(matrix.rows))
    while remaining:
        sink = next((node for node in sorted(remaining)
                     if all(matrix[child, node] == 0 for child in remaining if child != node)), None)
        if sink is None:
            return False
        remaining.remove(sink)
    return True


def support(matrix):
    return {(parent, child) for child in range(matrix.rows) for parent in range(matrix.cols)
            if matrix[child, parent] != 0}


def check(task):
    record = runtime.read(ROOT / "conditions367_v1" / (task["identifier"] + ".json"))
    nodes = record["nodes"]
    size = nodes - 4
    a, c, b, k, ell, e, gap = (sp.Rational(record["parameters"][name])
                              for name in ("a", "c", "b", "k", "l", "e", "s"))
    downstream = rational_matrix(record["downstream"])
    expected = sp.zeros(size)
    if record["structure"] == "layered":
        for node in range(2, size):
            for parent in range(2 * (node // 2 - 1), 2 * (node // 2 - 1) + 2):
                expected[node, parent] = sp.Rational(1, 4)
    else:
        for node in range(1, size):
            expected[node, node - 1] = sp.Rational(1, 4)
            if node >= 3:
                expected[node, node - 3] = sp.Rational(1, 4)
    assert downstream == expected
    response = (sp.eye(size) - downstream).inv()
    inputs = sp.Matrix([sp.Rational(1, 64) if i < 2 else 0 for i in range(size)])
    propagation = response * inputs
    assert list(propagation) == [sp.Rational(value) for value in record["propagation"]]
    assert (propagation.T * propagation)[0] <= sp.Rational(1, 256)
    core_options = [
        [sp.Matrix([1, a, a * c, b * a * c + gap]), sp.Matrix([1, a, a * c, e * a * c])],
        [sp.Matrix([0, 1, c, b * c]), sp.Matrix([0, 1, k, e * k])],
        [sp.Matrix([0, 0, 1, b]), sp.Matrix([0, ell, 1, e])],
        [sp.Matrix([0, 0, 0, 1])],
    ]

    def full_response(core):
        top = core.row_join(sp.zeros(4, size))
        bottom = (propagation * core[3, :]).row_join(response)
        return top.col_join(bottom)

    candidates = [[sp.Matrix([sp.Rational(value) for value in column]) for column in options]
                  for options in record["candidates"]]
    for target in range(4):
        for choice, column in enumerate(core_options[target]):
            assert candidates[target][choice] == column.col_join(propagation * column[3])
    for target in range(4, nodes):
        assert candidates[target] == [sp.zeros(4, 1).col_join(response[:, target - 4])]

    admitted, inverses = [], 0
    for choices in product((0, 1), repeat=3):
        core = sp.Matrix.hstack(*(core_options[j][choices[j]] for j in range(3)), core_options[3][0])
        full = full_response(core)
        if not acyclic_columns(full):
            continue
        coefficients = sp.eye(nodes) - full.inv()
        inverses += 1
        assert coefficients[4:, 4:] == downstream
        assert coefficients[4:, :4] == inputs * sp.Matrix([[0, 0, 0, 1]])
        assert all(coefficients[i, j] == 0 for i in range(4) for j in range(4, nodes))
        if sum(value != 0 for value in coefficients[3, :]) > 2:
            continue
        admitted.append((list(choices), coefficients[3, 0], support(coefficients)))
    saved = [(row["choices"], sp.Rational(row["shortcut"]), set(map(tuple, row["support"])))
             for row in record["admitted"]]
    assert admitted == saved and len(admitted) == 2
    assert len({tuple(sorted(row[2])) for row in admitted}) == (1 if gap else 2)
    assert set(map(tuple, record["physical_support"])) in [row[2] for row in admitted]

    witness_count = 0
    for witness in record["omitted_target_witnesses"]:
        core_coefficients = rational_matrix(witness["core"])
        core = (sp.eye(4) - core_coefficients).inv()
        full = full_response(core)
        coefficients = sp.eye(nodes) - full.inv()
        inverses += 1
        assert coefficients[3, 0] == 0 and sum(value != 0 for value in coefficients[3, :]) <= 2
        assert acyclic_columns(full)
        assert coefficients[4:, :4] == inputs * sp.Matrix([[0, 0, 0, 1]])
        assert coefficients[4:, 4:] == downstream
        for target in range(nodes):
            if target != witness["omitted"]:
                assert full[:, target] in candidates[target]
        witness_count += 1

    feedback_core = sp.zeros(4)
    feedback_core[1, 0], feedback_core[1, 2] = a * (1 - ell * c), ell
    feedback_core[2, 0], feedback_core[2, 1], feedback_core[3, 2] = a * (c - k), k, e
    feedback = feedback_core.row_join(sp.zeros(4, size)).col_join(
        (inputs * sp.Matrix([[0, 0, 0, 1]])).row_join(downstream))
    variable = sp.Symbol("z")
    assert sp.expand(feedback.charpoly(variable).as_expr() - variable ** (nodes - 2) * (variable ** 2 - k * ell)) == 0
    feedback_response = (sp.eye(nodes) - feedback).inv()
    inverses += 1
    for target in range(3):
        normalized = feedback_response[:, target] / feedback_response[target, target]
        assert normalized == candidates[target][1]
    for target in range(3, nodes):
        assert feedback_response[:, target] == candidates[target][0]
    assert feedback[3, 0] == 0 and 0 < k * ell < 1
    assert all(row["feasible"] == (gap == 0) and not row["singular_fallback"]
               for row in record["parent_domain_queries"])
    return dict(status="completed", symbolic_full_inverses=inverses,
                connected_conditions=1, omitted_target_witnesses=witness_count,
                complete_support_selections=len(admitted), stable_feedback_models=1,
                parent_domain_decisions=len(record["parent_domain_queries"]))
