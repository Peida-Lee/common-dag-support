"""Exact response-factor candidates and qualified common-DAG feasibility."""

from collections import deque
from fractions import Fraction
from itertools import combinations, product
from math import isqrt


def identity(size):
    return [[Fraction(i == j) for j in range(size)] for i in range(size)]


def inverse(matrix):
    size = len(matrix)
    rows = [list(row) + unit for row, unit in zip(matrix, identity(size))]
    for column in range(size):
        pivot = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("Singular principal block")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for i in range(size):
            if i != column and rows[i][column]:
                multiplier = rows[i][column]
                rows[i] = [a - multiplier * b for a, b in zip(rows[i], rows[column])]
    return [row[size:] for row in rows]


def response_from_lower(coefficients):
    size = len(coefficients)
    response = identity(size)
    for i, row in enumerate(coefficients):
        for parent, coefficient in enumerate(row[:i]):
            if coefficient:
                response[i] = [a + coefficient * b for a, b in zip(response[i], response[parent])]
    return response


def coefficients_from_columns(columns):
    size = len(columns)
    matrix = [[columns[j][i] for j in range(size)] for i in range(size)]
    structural = inverse(matrix)
    return [[Fraction(i == j) - structural[i][j] for j in range(size)] for i in range(size)]


def contrast(first, second):
    return [[a * b + c * d for b, d in zip(second, first)] for a, c in zip(first, second)]


def rational_sqrt(value):
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        raise ValueError("This exact fixture requires rational factor coordinates")
    return Fraction(numerator, denominator)


def factor_candidates(matrix, target):
    size = len(matrix)
    if not any(any(row) for row in matrix):
        raise ValueError("An informative target contrast is required")
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)):
        raise ValueError("A symmetric contrast is required")
    anchor = matrix[target][target]
    source = [row[target] for row in matrix]
    if anchor:
        residual = [[source[i] * source[k] - anchor * matrix[i][k]
                     for k in range(size)] for i in range(size)]
        pivot = next((i for i in range(size) if residual[i][i] > 0), None)
        if pivot is None:
            if any(any(row) for row in residual):
                raise ValueError("Incompatible factor Gram matrix")
            factors = [[value / anchor for value in source]]
        else:
            positive = rational_sqrt(residual[pivot][pivot])
            vector = [residual[i][pivot] / positive for i in range(size)]
            if any(residual[i][k] != vector[i] * vector[k]
                   for i in range(size) for k in range(size)):
                raise ValueError("Factor Gram matrix must have rank one")
            factors = [[(a + b) / anchor for a, b in zip(source, vector)],
                       [(a - b) / anchor for a, b in zip(source, vector)]]
    else:
        pivot = next((i for i, value in enumerate(source) if value), None)
        if pivot is None:
            raise ValueError("A zero anchor requires a nonzero target column")
        pivot_value = matrix[pivot][pivot] / (2 * source[pivot])
        factors = [[(matrix[i][pivot] - pivot_value * source[i]) / source[pivot]
                    for i in range(size)]]
    output = sorted(set(tuple(column) for column in factors))
    for column in output:
        partner = [s - anchor * a / 2 for s, a in zip(source, column)]
        if column[target] != 1 or contrast(column, partner) != matrix:
            raise ValueError("The complete target contrast is incompatible")
    return output


def support_masks(candidates):
    return [[sum(1 << i for i, value in enumerate(column) if i != j and value)
             for column in options] for j, options in enumerate(candidates)]


def horn_masks(domains):
    size = len(domains)
    if any(not options for options in domains):
        return dict(feasible=False, selection=None, order=[], support_entries=0, updates=0)
    owners, choices, remaining, reverse = [], [], [], [[] for _ in range(size)]
    support_entries = 0
    ready = deque()
    for j, options in enumerate(domains):
        for option, mask in enumerate(options):
            identifier = len(owners)
            owners.append(j)
            choices.append(option)
            remaining.append(mask.bit_count())
            if not mask:
                ready.append(identifier)
            while mask:
                bit = mask & -mask
                reverse[bit.bit_length() - 1].append(identifier)
                support_entries += 1
                mask -= bit
    selected, order, updates = [None] * size, [], 0
    while ready:
        identifier = ready.popleft()
        node = owners[identifier]
        if selected[node] is not None:
            continue
        selected[node] = choices[identifier]
        order.append(node)
        for neighbor in reverse[node]:
            remaining[neighbor] -= 1
            updates += 1
            if remaining[neighbor] == 0:
                ready.append(neighbor)
    feasible = len(order) == size
    return dict(feasible=feasible, selection=selected if feasible else None,
                order=order, support_entries=support_entries, updates=updates)


def horn(candidates):
    result = horn_masks(support_masks(candidates))
    if result["feasible"]:
        result["columns"] = [options[choice] for options, choice in zip(candidates, result["selection"])]
    return result


def acyclic_masks(masks):
    """Independent topological deletion on one fully selected graph."""
    size = len(masks)
    indegree = [0] * size
    for mask in masks:
        for i in range(size):
            indegree[i] += bool(mask & (1 << i))
    ready = [i for i, count in enumerate(indegree) if count == 0]
    removed = 0
    while ready:
        node = ready.pop()
        removed += 1
        for i in range(size):
            if masks[node] & (1 << i):
                indegree[i] -= 1
                if indegree[i] == 0:
                    ready.append(i)
    return removed == size


def exhaustive_masks(domains):
    return any(acyclic_masks(selection) for selection in product(*domains))


def parent_query(candidates, response, parents, roots=()):
    parents = tuple(parents)
    attempted = 0
    for choices in product(*(candidates[parent] for parent in parents)):
        block = [[column[i] for column in choices] for i in parents]
        try:
            block_inverse = inverse(block)
        except ValueError:
            continue
        coefficients = tuple(sum((choices[i][response] * block_inverse[i][j]
                                  for i in range(len(parents))), Fraction(0))
                             for j in range(len(parents)))
        filtered = []
        for j, options in enumerate(candidates):
            filtered.append([column for column in options
                             if all(column[root] == (j == root) for root in roots)
                             and column[response] - sum((value * column[parent]
                                 for parent, value in zip(parents, coefficients)), Fraction(0))
                             == (j == response)])
        attempted += 1
        result = horn(filtered)
        if result["feasible"]:
            return dict(feasible=True, columns=result["columns"], parents=parents,
                        coefficients=coefficients, branches=attempted)
    return dict(feasible=False, branches=attempted)


def local_indegree_query(candidates, source, response, maximum, roots=()):
    available = [i for i in range(len(candidates)) if i not in (source, response)]
    attempted = 0
    for size in range(maximum + 1):
        for parents in combinations(available, size):
            result = parent_query(candidates, response, parents, roots)
            attempted += result["branches"]
            if result["feasible"]:
                result["branches"] = attempted
                return result
    return dict(feasible=False, branches=attempted)


def exhaustive_query(candidates, source, response, parents=None, maximum=None, roots=()):
    domains = support_masks(candidates)
    for indices in product(*(range(len(options)) for options in candidates)):
        masks = [options[index] for options, index in zip(domains, indices)]
        if not acyclic_masks(masks):
            continue
        columns = [options[index] for options, index in zip(candidates, indices)]
        coefficients = coefficients_from_columns(columns)
        if any(any(coefficients[root]) for root in roots) or coefficients[response][source]:
            continue
        actual_parents = {i for i, value in enumerate(coefficients[response]) if value}
        if parents is not None and not actual_parents.issubset(parents):
            continue
        if maximum is not None and len(actual_parents) > maximum:
            continue
        return True
    return False
