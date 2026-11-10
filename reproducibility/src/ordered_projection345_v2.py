"""Select a fixed null-orbit transformation using exact pooled outcome energy."""

from fractions import Fraction
from itertools import combinations

import numpy as np

from ordered_projection345_v1 import PRESSURES, audit as original_audit, evaluate as original_evaluate
from exact_randomization_KS323_v1 import test as exact_ks

METHODS = ("Raw_KS", "Ordered_center_KS", "Combined_KS", "Selected_KS")


def select_projection(values, minimum_width=50):
    ratios = [float(value).as_integer_ratio() for value in values]
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    integers = [numerator << (exponent - denominator.bit_length() + 1)
                for numerator, denominator in ratios]
    n = len(integers)
    groups = np.array_split(np.arange(n), max(1, n // minimum_width))
    raw_energy = Fraction(n * sum(value * value for value in integers) - sum(integers) ** 2, n)
    projected_energy = Fraction(0)
    for group in groups:
        block = [integers[int(index)] for index in group]
        projected_energy += Fraction(len(block) * sum(value * value for value in block) - sum(block) ** 2, len(block))
    retention = Fraction(n - len(groups), n - 1)
    selected = projected_energy < retention ** 2 * raw_energy
    return dict(selected="Ordered_center_KS" if selected else "Raw_KS",
                exact_raw_energy=str(raw_energy), exact_centered_energy=str(projected_energy),
                expected_contrast_retention=str(retention), assignment_used=False,
                target_values_used=False, exact_energy_comparison=True)


def evaluate(assignment, values, family_queries, adjust=True):
    result = original_evaluate(assignment, values, family_queries, adjust)
    if adjust:
        selection = select_projection(values)
        result["selection"] = selection
        result["methods"]["Selected_KS"] = dict(result["methods"][selection["selected"]],
                                                   selected_component=selection["selected"])
    else:
        result["methods"]["Selected_KS"] = dict(result["methods"]["Raw_KS"], projection_applied=False)
    return result


def audit():
    from ordered_projection345_v1 import centered_ranks

    result = original_audit()
    checked = 0
    for values in (np.arange(8, dtype=float), np.array([0, 0, 1, 1, 9, 9, 10, 10], dtype=float), np.ones(8)):
        selection = select_projection(values, minimum_width=3)
        ranks, _ = centered_ranks(values, minimum_width=3)
        selected_values = ranks if selection["selected"] == "Ordered_center_KS" else values
        probabilities = []
        for indices in combinations(range(8), 4):
            assignment = np.zeros(8, dtype=int)
            assignment[list(indices)] = 1
            assert select_projection(values, minimum_width=3) == selection
            probabilities.append(Fraction(exact_ks(assignment, selected_values, 1)["raw_P_exact"]))
        for level in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10), Fraction(1, 2)):
            assert Fraction(sum(p <= level for p in probabilities), len(probabilities)) <= level
            checked += 1
    result["outcome_only_selection_superuniform_checks"] = checked
    result["version"] = "v2"
    return result
