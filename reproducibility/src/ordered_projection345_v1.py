"""Assignment-blind ordered centering with exact residual rank and tie ordering."""

from fractions import Fraction
from itertools import combinations
from math import lcm
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "support"))
if (ROOT.parent / "round323").exists():
    sys.path.append(str(ROOT.parent / "round323"))
from exact_randomization_KS323_v1 import test as exact_ks

METHODS = ("Raw_KS", "Ordered_center_KS", "Combined_KS")
PRESSURES = {"pressure_upwind", "pressure_intake", "pressure_ambient", "pressure_downwind"}


def centered_ranks(values, minimum_width=50):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    assert len(values) >= 2 and minimum_width >= 1
    ratio = [float(value).as_integer_ratio() for value in values]
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratio)
    integers = [numerator << (exponent - denominator.bit_length() + 1)
                for numerator, denominator in ratio]
    groups = np.array_split(np.arange(len(values)), max(1, len(values) // minimum_width))
    denominator = lcm(*(len(group) for group in groups))
    residual = [0] * len(values)
    for group in groups:
        total = sum(integers[int(index)] for index in group)
        width = len(group)
        scale = denominator // width
        for index in group:
            residual[int(index)] = scale * (width * integers[int(index)] - total)
    order = sorted(range(len(values)), key=residual.__getitem__)
    ranks = np.empty(len(values), dtype=np.int64)
    rank = 0
    for position, index in enumerate(order):
        if position and residual[index] != residual[order[position - 1]]:
            rank += 1
        ranks[index] = rank
    return ranks, dict(blocks=len(groups), block_sizes=[len(group) for group in groups],
                       minimum_width=minimum_width, source_dyadic_exponent=exponent,
                       common_block_denominator=denominator, distinct_residual_values=rank + 1,
                       exact_integer_order=True, assignment_used=False, all_rows_retained=True)


def combined(raw, adjusted):
    probability = min(Fraction(1), 2 * min(Fraction(raw["raw_P_exact"]), Fraction(adjusted["raw_P_exact"])))
    threshold = Fraction(raw["raw_P_threshold_exact"])
    return dict(raw_P_exact=str(probability), raw_P_float=float(probability),
                raw_P_threshold_exact=str(threshold), rejected=probability <= threshold,
                family_queries=raw["family_queries"], family_alpha=raw["family_alpha"],
                components=[raw["raw_P_exact"], adjusted["raw_P_exact"]],
                multiplicity="two-component union bound within the original query family")


def evaluate(assignment, values, family_queries, adjust=True):
    raw = exact_ks(assignment, values, family_queries)
    if not adjust:
        return dict(methods={"Raw_KS": raw, "Combined_KS": dict(raw, projection_applied=False)})
    ranks, metadata = centered_ranks(values)
    adjusted = exact_ks(assignment, ranks, family_queries)
    return dict(methods={"Raw_KS": raw, "Ordered_center_KS": adjusted,
                         "Combined_KS": combined(raw, adjusted)}, projection=metadata)


def audit():
    values = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2])
    ranks, metadata = centered_ranks(values, minimum_width=3)
    exact = [Fraction.from_float(float(value)) for value in values]
    residual = [Fraction(0)] * len(values)
    for group in np.array_split(np.arange(len(values)), 2):
        mean = sum(exact[int(index)] for index in group) / len(group)
        for index in group:
            residual[int(index)] = exact[int(index)] - mean
    comparisons = 0
    for i in range(len(values)):
        for j in range(len(values)):
            assert bool(ranks[i] < ranks[j]) == (residual[i] < residual[j])
            assert bool(ranks[i] == ranks[j]) == (residual[i] == residual[j])
            comparisons += 2
    centered = [value - sum(exact) / len(exact) for value in exact]
    assert sum(value * value for value in residual) <= sum(value * value for value in centered)
    contrasts = []
    for selected in combinations(range(8), 4):
        contrasts.append(sum(residual[index] for index in selected))
    assert sum(contrasts) == 0
    variance = sum(value * value for value in contrasts) / len(contrasts)
    assert variance == Fraction(4 * 4, 8 * 7) * sum(value * value for value in residual)
    panels = (np.arange(8, dtype=float), np.array([0, 0, 1, 1, 9, 9, 10, 10], dtype=float),
              np.ones(8))
    conditional_checks = 0
    case_count = 0
    for outcomes in panels:
        adjusted_ranks, _ = centered_ranks(outcomes, minimum_width=3)
        p_values = []
        for selected in combinations(range(8), 4):
            assignment = np.zeros(8, dtype=int)
            assignment[list(selected)] = 1
            raw = exact_ks(assignment, outcomes, 1)
            adjusted = exact_ks(assignment, adjusted_ranks, 1)
            p_values.append([Fraction(raw["raw_P_exact"]), Fraction(adjusted["raw_P_exact"]),
                             Fraction(combined(raw, adjusted)["raw_P_exact"])])
            case_count += 1
        for method in range(3):
            for level in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10), Fraction(1, 2)):
                assert Fraction(sum(row[method] <= level for row in p_values), len(p_values)) <= level
                conditional_checks += 1
    single, _ = centered_ranks(values, minimum_width=50)
    for assignment_tuple in combinations(range(8), 4):
        assignment = np.zeros(8, dtype=int)
        assignment[list(assignment_tuple)] = 1
        assert exact_ks(assignment, values, 1)["raw_P_exact"] == exact_ks(assignment, single, 1)["raw_P_exact"]
    return dict(status="passed", exact_fraction_rank_comparisons=comparisons,
                small_fixed_count_cases=case_count, conditional_superuniform_checks=conditional_checks,
                single_block_parity_cases=70, ties_preserved=True,
                assignment_blind=True, exact_variance_and_contraction_checks=3,
                example_projection=metadata)
