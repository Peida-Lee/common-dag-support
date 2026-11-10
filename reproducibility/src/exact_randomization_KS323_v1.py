"""Exact conditional KS bounds with integer path counting and tied observations."""

from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from math import comb

import numpy as np


def test(assignment, values, query_count, alpha=.05):
    assignment = np.asarray(assignment)
    values = np.asarray(values, dtype=float)
    if assignment.ndim != 1 or values.ndim != 1 or len(assignment) != len(values) or len(values) < 2:
        raise ValueError("Aligned assignment and response vectors are required")
    if not np.isfinite(values).all() or not np.isin(assignment, [0, 1]).all():
        raise ValueError("Finite readouts and binary assignments are required")
    if not isinstance(query_count, int) or query_count < 1 or not 0 < alpha < 1:
        raise ValueError("A positive fixed family size and error budget are required")
    n = len(values)
    n1, n0 = int(np.sum(assignment)), int(n - np.sum(assignment))
    tail, total, height = 1, 1, 0
    if n1 and n0:
        order = np.argsort(values, kind="stable")
        ranked = values[order]
        treatment = np.cumsum(assignment[order].astype(np.int64))
        control = np.arange(1, n + 1) - treatment
        ends = np.r_[ranked[:-1] != ranked[1:], True]
        height = int(np.max(np.abs(treatment[ends] * n0 - control[ends] * n1)))
        if height:
            # Symmetry keeps the single dynamic-programming row short.
            n1, n0 = max(n1, n0), min(n1, n0)
            counts = [0] * (n0 + 1)
            counts[0] = 1
            for i in range(n1 + 1):
                lower = max(0, (i * n0 - height) // n1 + 1)
                upper = min(n0, (i * n0 + height - 1) // n1)
                left = 0
                for j in range(lower, upper + 1):
                    value = counts[j] + left
                    counts[j] = value
                    left = value
            total = comb(n, n1)
            tail = total - counts[n0]
            if not 0 < tail <= total:
                raise ArithmeticError("Invalid exact KS path count")
    probability = Fraction(tail, total)
    threshold = Fraction(str(alpha)) / query_count
    with localcontext() as context:
        context.prec = 45
        context.rounding = ROUND_CEILING
        decimal_upper = Decimal(tail) / Decimal(total)
    return {"n": n, "counts": [int(np.sum(assignment == label)) for label in (0, 1)],
            "KS_height_integer": height,
            "KS_statistic_exact": str(Fraction(height, n1 * n0)) if n1 and n0 else "0",
            "raw_P_exact": str(probability), "raw_P_decimal_upper": str(decimal_upper),
            "raw_component_P": [float(decimal_upper)],
            "raw_P_threshold_exact": str(threshold), "family_queries": query_count,
            "family_alpha": str(Fraction(str(alpha))),
            "rejected": probability <= threshold, "integer_calibration": True,
            "ties": "continuous_rank_tail_conservative_under_ties"}


def fixture():
    from scipy.stats import ks_2samp
    import warnings
    n = 10
    assignments = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(int)
    records = []
    fallbacks = 0
    for tied in (False, True):
        values = np.arange(n, dtype=float) // 3 if tied else np.arange(n, dtype=float)
        by_count = {count: [] for count in range(1, n)}
        for assignment in assignments:
            count = int(np.sum(assignment))
            if count in by_count:
                result = test(assignment, values, 1)
                by_count[count].append(Fraction(result["raw_P_exact"]))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    scipy_p = ks_2samp(values[assignment == 0], values[assignment == 1], method="exact").pvalue
                fallbacks += sum("Exact calculation unsuccessful" in str(item.message) for item in caught)
                assert abs(float(Fraction(result["raw_P_exact"])) - scipy_p) < 2e-12
        for count, probabilities in by_count.items():
            for level in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10), Fraction(1, 2)):
                rejected = sum(value <= level for value in probabilities)
                assert Fraction(rejected, len(probabilities)) <= level
            records.append({"tied": tied, "treatment_count": count,
                            "assignments": len(probabilities), "checks": "passed"})
    assert test(np.zeros(n), np.arange(n), 1)["raw_P_exact"] == "1"
    assert test(assignments[123], np.ones(n), 1)["raw_P_exact"] == "1"
    return {"status": "passed", "assignment_vectors": len(assignments), "cases": records,
            "SciPy_parity_checks": sum(record["assignments"] for record in records),
            "SciPy_reported_fallbacks": fallbacks,
            "constant_and_single_arm_checks": 2}
