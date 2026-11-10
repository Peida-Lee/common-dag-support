"""Conservative factor regions and common missing-edge branch exclusion."""

from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round356"))
from response_factors356_v1 import horn_masks
from outward_covariance359_v1 import Interval, universal


def factor_regions(contrast, target):
    size = len(contrast)
    anchor = contrast[target][target]
    if anchor.contains(0) or not anchor.finite():
        column = [universal() for _ in range(size)]
        column[target] = Interval.point(1)
        return [column], dict(status="unresolved_anchor", pivot=None, regions=1)
    source = [row[target] for row in contrast]
    gram = [[source[i]*source[k]-anchor*contrast[i][k] for k in range(size)] for i in range(size)]
    if any(gram[i][i].upper < 0 for i in range(size)):
        return None, dict(status="qualification_failure", reason="negative_factor_Gram_diagonal")
    pivots = [i for i in range(size) if i != target and gram[i][i].lower > 0]
    if pivots:
        pivot = max(pivots, key=lambda i: gram[i][i].lower)
        root = gram[pivot][pivot].sqrt()
        factor = [gram[i][pivot]/root for i in range(size)]
        factor[pivot] = root
        factor[target] = Interval.point(0)
        columns = [[(value+sign*other)/anchor for value, other in zip(source, factor)] for sign in (1, -1)]
        status = "separated_factors"
    else:
        pivot = None
        factor = []
        for i in range(size):
            radius = Interval(0., max(0., gram[i][i].upper)).sqrt().upper
            factor.append(Interval(-radius, radius))
        factor[target] = Interval.point(0)
        columns = [[(value+other)/anchor for value, other in zip(source, factor)]]
        status = "merged_or_unseparated_factors"
    for column in columns:
        column[target] = Interval.point(1)
    return columns, dict(status=status, pivot=pivot, regions=len(columns), anchor=anchor.pair())


def coefficient_region(choices, parents, targets, response):
    dimension = len(parents)
    right = [column[response]-(target == response) for column, target in zip(choices, targets)]
    if dimension == 0:
        return []
    if dimension == 1:
        divisor = choices[0][parents[0]]
        return None if divisor.contains(0) or not divisor.finite() else [right[0]/divisor]
    if dimension != 2:
        raise ValueError("The numerical implementation supports up to two remaining parents")
    a, b = choices[0][parents[0]], choices[1][parents[0]]
    c, d = choices[0][parents[1]], choices[1][parents[1]]
    determinant = a*d-b*c
    if determinant.contains(0) or not determinant.finite():
        return None
    return [(right[0]*d-right[1]*c)/determinant,
            (right[1]*a-right[0]*b)/determinant]


def root_filter(domains, roots):
    output = []
    for index, options in enumerate(domains):
        if options is None:
            output.append(None)
            continue
        kept = []
        for column in options:
            if all(column[root].contains(index == root) for root in roots):
                restricted = list(column)
                for root in roots:
                    restricted[root] = Interval.point(index == root)
                kept.append(restricted)
        output.append(kept)
    return output


def missing_query(domains, response, parents, targets, roots=()):
    parents, targets = tuple(parents), tuple(targets)
    if len(parents) != len(targets) or len(parents) > 2:
        raise ValueError("One known spanning target per remaining parent is required")
    if response in parents or response in roots or len(set(parents)) != len(parents):
        raise ValueError("The qualified response, parent and root indices are incompatible")
    if any(domains[target] is None for target in targets):
        return dict(reported=False, unresolved=True, reason="unobserved_spanning_target", branches=[])
    qualified = root_filter(domains, roots)
    if any(options == [] for options in qualified):
        return dict(reported=False, unresolved=True, reason="root_qualification_failure", branches=[])
    branches = []
    for choices in product(*(qualified[target] for target in targets)):
        selected_targets = dict(zip(targets, choices))
        preliminary = []
        for index, options in enumerate(qualified):
            if options is None:
                preliminary.append([0])
            else:
                fixed = [selected_targets[index]] if index in selected_targets else options
                preliminary.append([sum(1 << i for i, value in enumerate(column)
                                         if i != index and value.excludes_zero()) for column in fixed])
        preliminary_result = horn_masks(preliminary)
        if not preliminary_result["feasible"]:
            branches.append(dict(excluded=True, reason="common_DAG_cycle_before_inverse",
                                 mandatory_supports=preliminary))
            continue
        coefficients = coefficient_region(choices, parents, targets, response)
        if coefficients is None or any(not value.finite() for value in coefficients):
            branches.append(dict(excluded=False, reason="uncertain_spanning_inverse"))
            continue
        supports, option_counts = [], []
        for index, options in enumerate(qualified):
            if options is None:
                mask = 0
                if index in parents and coefficients[parents.index(index)].excludes_zero():
                    mask = 1 << response
                supports.append([mask])
                option_counts.append(1)
                continue
            retained = []
            for column in options:
                residual = column[response]-int(index == response)
                for parent, coefficient in zip(parents, coefficients):
                    residual = residual-coefficient*column[parent]
                if residual.contains(0):
                    retained.append(sum(1 << i for i, value in enumerate(column)
                                        if i != index and value.excludes_zero()))
            supports.append(retained)
            option_counts.append(len(retained))
        result = horn_masks(supports)
        reason = "relaxed_acyclic_selection" if result["feasible"] else (
                 "empty_row_domain" if 0 in option_counts else "common_DAG_cycle")
        branches.append(dict(excluded=not result["feasible"], reason=reason,
                             coefficients=[value.pair() for value in coefficients],
                             option_counts=option_counts, mandatory_supports=supports,
                             horn_order=result["order"]))
    reported = bool(branches) and all(branch["excluded"] for branch in branches)
    return dict(reported=reported, unresolved=not reported,
                reason="all_missing_branches_excluded" if reported else "retained_missing_outer_branch",
                branches=branches)
