"""Exact affine-flat queries for partially targeted response columns."""

from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round356"))
import response_factors356_v1 as common


def affine_flat(equations, dimension):
    rows = [list(coefficients) + [right] for coefficients, right in equations]
    pivot_columns = []
    rank = 0
    for column in range(dimension):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [value / divisor for value in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][column]:
                multiplier = rows[i][column]
                rows[i] = [a-multiplier*b for a, b in zip(rows[i], rows[rank])]
        pivot_columns.append(column)
        rank += 1
    if any(not any(row[:dimension]) and row[-1] for row in rows):
        return None
    particular = [Fraction(0)] * dimension
    for i, pivot in enumerate(pivot_columns):
        particular[pivot] = rows[i][-1]
    basis = []
    for free in range(dimension):
        if free in pivot_columns:
            continue
        vector = [Fraction(i == free) for i in range(dimension)]
        for i, pivot in enumerate(pivot_columns):
            vector[pivot] = -rows[i][free]
        basis.append(tuple(vector))
    return dict(particular=tuple(particular), basis=tuple(basis),
                key=tuple(tuple(row) for row in rows[:rank]))


def dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Fraction(0))


def contains_equation(flat, equation):
    coefficients, right = equation
    return dot(coefficients, flat["particular"]) == right and all(
        dot(coefficients, vector) == 0 for vector in flat["basis"])


def forced_zero(flat, coordinate):
    return flat["particular"][coordinate] == 0 and all(
        vector[coordinate] == 0 for vector in flat["basis"])


def generic_point(flat, required_nonzero):
    basis = flat["basis"]
    attempts = len(required_nonzero)*len(basis) + 1
    for value in range(attempts):
        point = tuple(start + sum((Fraction(value)**(i+1)*vector[j]
                      for i, vector in enumerate(basis)), Fraction(0))
                      for j, start in enumerate(flat["particular"]))
        if all(point[j] for j in required_nonzero):
            return point
    raise ArithmeticError("The finite hyperplane witness bound was violated")


def column_equation(column, index, response, parents):
    return tuple(column[parent] for parent in parents), column[response] - (index == response)


def canonical_columns(domains, selected, response, parents, coefficients):
    size = len(domains)
    parent_index = {parent: i for i, parent in enumerate(parents)}
    columns = []
    for j, options in enumerate(domains):
        if options is not None:
            columns.append(options[selected[j]])
            continue
        column = [Fraction(i == j) for i in range(size)]
        if j in parent_index:
            column[response] = coefficients[parent_index[j]]
        columns.append(tuple(column))
    return columns


def check_witness(columns, response, parents, coefficients, roots):
    masks = [options[0] for options in common.support_masks([[column] for column in columns])]
    assert common.acyclic_masks(masks)
    for j, column in enumerate(columns):
        assert column[j] == 1
        assert all(column[root] == (j == root) for root in roots)
        assert column[response] - dot(coefficients, [column[p] for p in parents]) == (j == response)


def normalized_equation(equation):
    coefficients, right = equation
    leading = next((value for value in coefficients if value), None)
    if leading is None:
        return None
    return tuple(value/leading for value in coefficients), right/leading


def spanning_query(domains, response, parents, targets, roots=()):
    parents, targets = tuple(parents), tuple(targets)
    if response in parents or response in roots or len(set(parents)) != len(parents):
        raise ValueError("The qualified response, parents and root rows must be distinct as declared")
    if len(targets) != len(parents) or len(set(targets)) != len(targets):
        raise ValueError("A spanning target set has one distinct column per possible parent")
    if any(domains[target] is None for target in targets):
        raise ValueError("Spanning target columns require finite candidate sets")
    qualified = [None if options is None else [column for column in options
                 if all(column[root] == (j == root) for root in roots)]
                 for j, options in enumerate(domains)]
    checked = 0
    for choices in product(*(qualified[target] for target in targets)):
        block = [[column[parent] for column in choices] for parent in parents]
        try:
            block_inverse = common.inverse(block)
        except ValueError:
            result = parent_query(domains, response, parents, roots, use_known_parent_shortcut=False)
            result["spanning_singular_fallback"] = True
            return result
        right = [column[response] - (target == response) for target, column in zip(targets, choices)]
        coefficients = tuple(sum((right[i]*block_inverse[i][j]
                                  for i in range(len(parents))), Fraction(0))
                             for j in range(len(parents)))
        filtered = []
        for j, options in enumerate(qualified):
            if options is None:
                filtered.append(None)
            else:
                filtered.append([column for column in options if
                    dot(column_equation(column, j, response, parents)[0], coefficients)
                    == column_equation(column, j, response, parents)[1]])
        masks = []
        for j, options in enumerate(filtered):
            if options is None:
                mask = (1 << response) if j in parents and coefficients[parents.index(j)] else 0
                masks.append([mask])
            else:
                masks.append([sum(1 << i for i, value in enumerate(column) if i != j and value)
                              for column in options])
        selection = common.horn_masks(masks)
        checked += 1
        if selection["feasible"]:
            columns = canonical_columns(filtered, selection["selection"], response, parents, coefficients)
            check_witness(columns, response, parents, coefficients, roots)
            return dict(feasible=True, columns=columns, coefficients=coefficients, parents=parents,
                        branches=checked, solver="spanning_targets", spanning_singular_fallback=False)
    return dict(feasible=False, branches=checked, solver="spanning_targets",
                spanning_singular_fallback=False)


def parent_query(domains, response, parents, roots=(), use_known_parent_shortcut=True):
    parents = tuple(parents)
    size, dimension = len(domains), len(parents)
    if response in parents or response in roots or len(set(parents)) != dimension:
        raise ValueError("The qualified response, parents and root rows must be distinct as declared")
    if use_known_parent_shortcut and all(domains[p] is not None for p in parents):
        completed = [options if options is not None else [tuple(Fraction(i == j) for i in range(size))]
                     for j, options in enumerate(domains)]
        result = common.parent_query(completed, response, parents, roots)
        result.update(solver="known_parent_columns", flats_checked=0)
        return result
    qualified, equations = [], []
    collection = set()
    for j, options in enumerate(domains):
        if options is None:
            qualified.append(None)
            equations.append(None)
            continue
        kept, restrictions = [], []
        for column in options:
            if any(column[root] != (j == root) for root in roots):
                continue
            equation = column_equation(column, j, response, parents)
            if not any(equation[0]) and equation[1]:
                continue
            kept.append(column)
            restrictions.append(equation)
            normalized = normalized_equation(equation)
            if normalized is not None:
                collection.add(normalized)
        if not kept:
            return dict(feasible=False, flats_checked=0, solver="affine_flats", branches=0)
        qualified.append(kept)
        equations.append(restrictions)
    unknown_parent_indices = [i for i, p in enumerate(parents) if domains[p] is None]
    for coordinate in unknown_parent_indices:
        collection.add((tuple(Fraction(i == coordinate) for i in range(dimension)), Fraction(0)))
    collection = sorted(collection)
    seen = set()
    checked = 0
    for rank in range(min(dimension, len(collection)) + 1):
        for selected_equations in combinations(collection, rank):
            flat = affine_flat(selected_equations, dimension)
            if flat is None or flat["key"] in seen:
                continue
            seen.add(flat["key"])
            filtered, masks = [], []
            active_unknown = [i for i in unknown_parent_indices if not forced_zero(flat, i)]
            for j, options in enumerate(qualified):
                if options is None:
                    filtered.append(None)
                    masks.append([1 << response if j in parents and parents.index(j) in active_unknown else 0])
                    continue
                retained = [column for column, equation in zip(options, equations[j])
                            if contains_equation(flat, equation)]
                filtered.append(retained)
                masks.append([sum(1 << i for i, value in enumerate(column) if i != j and value)
                              for column in retained])
            checked += 1
            selection = common.horn_masks(masks)
            if not selection["feasible"]:
                continue
            point = generic_point(flat, active_unknown)
            columns = canonical_columns(filtered, selection["selection"], response, parents, point)
            check_witness(columns, response, parents, point, roots)
            return dict(feasible=True, columns=columns, coefficients=point, parents=parents,
                        flats_checked=checked, solver="affine_flats", branches=checked,
                        flat_dimension=len(flat["basis"]), equation_count=len(collection))
    return dict(feasible=False, flats_checked=checked, solver="affine_flats", branches=checked,
                equation_count=len(collection))


def minor_witness(equations, dimension, required_nonzero):
    """Independent zero-, one- and two-dimensional Cramer reference."""
    if dimension > 2:
        raise ValueError("The independent minor reference supports at most two parents")
    if dimension == 0:
        return () if all(right == 0 for _, right in equations) else None
    nonzero = [(coefficients, right) for coefficients, right in equations if any(coefficients)]
    if any(not any(coefficients) and right for coefficients, right in equations):
        return None
    if not nonzero:
        return tuple(Fraction(i in required_nonzero) for i in range(dimension))
    if dimension == 1:
        point = (nonzero[0][1]/nonzero[0][0][0],)
        return point if all(dot(row, point) == right for row, right in equations) and all(
            point[i] for i in required_nonzero) else None
    first, first_right = nonzero[0]
    for second, second_right in nonzero[1:]:
        determinant = first[0]*second[1] - first[1]*second[0]
        if determinant:
            point = ((first_right*second[1]-first[1]*second_right)/determinant,
                     (first[0]*second_right-first_right*second[0])/determinant)
            return point if all(dot(row, point) == right for row, right in equations) and all(
                point[i] for i in required_nonzero) else None
    for value in range(4):
        point = ((first_right-first[1]*value)/first[0], Fraction(value)) if first[0] else (
            Fraction(value), first_right/first[1])
        if all(dot(row, point) == right for row, right in equations) and all(
                point[i] for i in required_nonzero):
            return point
    return None


def exhaustive_query(domains, response, parents, roots=()):
    parents = tuple(parents)
    known = [j for j, options in enumerate(domains) if options is not None]
    unknown_parents = [i for i, p in enumerate(parents) if domains[p] is None]
    for choices in product(*(domains[j] for j in known)):
        chosen = dict(zip(known, choices))
        if any(column[root] != (j == root) for j, column in chosen.items() for root in roots):
            continue
        equations = [column_equation(column, j, response, parents) for j, column in chosen.items()]
        for active in product((False, True), repeat=len(unknown_parents)):
            required = [i for i, nonzero in zip(unknown_parents, active) if nonzero]
            zero_equations = [(tuple(Fraction(i == coordinate) for i in range(len(parents))), Fraction(0))
                              for coordinate, nonzero in zip(unknown_parents, active) if not nonzero]
            point = minor_witness(equations + zero_equations, len(parents), required)
            if point is None:
                continue
            selected_domains = [[chosen[j]] if j in chosen else None for j in range(len(domains))]
            columns = canonical_columns(selected_domains, [0]*len(domains), response, parents, point)
            masks = [options[0] for options in common.support_masks([[column] for column in columns])]
            if common.acyclic_masks(masks):
                check_witness(columns, response, parents, point, roots)
                return dict(feasible=True, columns=columns, coefficients=point)
    return dict(feasible=False)
