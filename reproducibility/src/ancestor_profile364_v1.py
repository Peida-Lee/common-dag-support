"""Ancestor qualification, Fieller inversion and full-line quadratic profiling."""

from fractions import Fraction
from math import isqrt
from pathlib import Path
import sys

ASSETS = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ASSETS / folder) for folder in ("round359", "round335")]
from outward_covariance359_v1 import Interval
from rank_calibration335_v1 import rank_decision

ALPHA_ANCESTOR_EDGE = Fraction(1, 300)
ALPHA_FIELLER = Fraction(1, 50)
ALPHA_VARIANCE_PAIR = Fraction(1, 100)


def endpoints(value):
    return Fraction.from_float(value.lower), Fraction.from_float(value.upper)


def symmetric_boxes(values):
    size = len(values)
    return [[Interval(max(float(values[i][j][0]), float(values[j][i][0])),
                      min(float(values[i][j][1]), float(values[j][i][1])))
             for j in range(size)] for i in range(size)]


def square(value):
    result = value*value
    return Interval(0., result.upper) if value.contains(0) else result


def cutoff(bank, alpha):
    size = len(bank["upper"])
    index = (size+1)*(1-alpha)
    rank = (index.numerator+index.denominator-1)//index.denominator
    return None if rank > size else Fraction(bank["upper"][rank-1])


def symmetric_ratio_interval(numerator, denominator):
    ratio = numerator/denominator
    if not ratio.finite() or ratio.lower <= 0:
        return None
    reciprocal = Interval.point(1)/ratio
    return max(Fraction(1), endpoints(ratio)[0], endpoints(reciprocal)[0]), max(endpoints(ratio)[1], endpoints(reciprocal)[1])


def ancestor_qualification(samples, bank):
    decisions = []
    for target, coordinate in ((0, 1), (1, 2), (2, 3)):
        statistic = symmetric_ratio_interval(samples[target+1][coordinate][coordinate], samples[0][coordinate][coordinate])
        if statistic is None:
            decisions.append(dict(status="numerically_unresolved", target=target, coordinate=coordinate))
            continue
        decision = rank_decision(statistic, list(map(Fraction, bank["lower"])), list(map(Fraction, bank["upper"])),
                                 ALPHA_ANCESTOR_EDGE)
        decisions.append(dict(**decision, target=target, coordinate=coordinate))
    return dict(qualified=all(item["status"] == "certified" for item in decisions), decisions=decisions)


def square_root_bounds(value, bits=80):
    value = Fraction(value)
    scale = 1 << bits
    integer = isqrt(value.numerator*scale*scale//value.denominator)
    lower = Fraction(integer, scale)
    upper = lower if lower*lower == value else Fraction(integer+1, scale)
    return lower, upper


def quadratic_sublevel(a, b, c):
    """Closed outer intervals for one exact rational quadratic sublevel set."""
    a, b, c = map(Fraction, (a, b, c))
    if a == 0:
        if b == 0:
            return [(None, None)] if c <= 0 else []
        root = -c/b
        return [(None, root)] if b > 0 else [(root, None)]
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return [] if a > 0 else [(None, None)]
    if discriminant == 0:
        root = -b/(2*a)
        return [(root, root)] if a > 0 else [(None, None)]
    root_lower, root_upper = square_root_bounds(discriminant)
    first = sorted(((-b-root_lower)/(2*a), (-b-root_upper)/(2*a)))
    second = sorted(((-b+root_lower)/(2*a), (-b+root_upper)/(2*a)))
    left, right = (first, second) if first[0] < second[0] else (second, first)
    return [(left[0], right[1])] if a > 0 else [(None, left[1]), (right[0], None)]


def intersection(first, second):
    output = []
    for left_a, right_a in first:
        for left_b, right_b in second:
            lower = left_b if left_a is None else left_a if left_b is None else max(left_a, left_b)
            upper = right_b if right_a is None else right_a if right_b is None else min(right_a, right_b)
            if lower is None or upper is None or lower <= upper:
                output.append((lower, upper))
    return merge(output)


def merge(intervals):
    ordered = sorted(intervals, key=lambda pair: (pair[0] is not None, pair[0] or Fraction(0)))
    output = []
    for lower, upper in ordered:
        if output and (output[-1][1] is None or lower is None or lower <= output[-1][1]):
            previous_lower, previous_upper = output.pop()
            output.append((previous_lower, None if previous_upper is None or upper is None else max(previous_upper, upper)))
        else:
            output.append((lower, upper))
    return output


def outer_sublevel(coefficients):
    constant, linear, quadratic = coefficients
    c_lower = endpoints(constant)[0]
    a_lower = endpoints(quadratic)[0]
    b_lower, b_upper = endpoints(linear)
    positive = intersection(quadratic_sublevel(a_lower, b_lower, c_lower), [(Fraction(0), None)])
    negative = intersection(quadratic_sublevel(a_lower, b_upper, c_lower), [(None, Fraction(0))])
    return merge(positive+negative)


def fieller_coefficients(samples, degrees, critical):
    slope_y, slope_v, sums = [], [], [Interval.point(0) for _ in range(3)]
    inverse_h = Interval.point(0)
    for cell in (0, 3):
        covariance = samples[cell]
        xx, xv, xy = covariance[1][1], covariance[1][2], covariance[1][3]
        if xx.lower <= 0:
            return None
        slope_y.append(xy/xx)
        slope_v.append(xv/xx)
        inverse_h += Interval.point(1)/(degrees*xx)
        residual = [covariance[3][3]-square(xy)/xx,
                    -2*covariance[2][3]+2*xy*xv/xx,
                    covariance[2][2]-square(xv)/xx]
        sums = [value+degrees*term for value, term in zip(sums, residual)]
    dy, dv = slope_y[0]-slope_y[1], slope_v[0]-slope_v[1]
    scale = Interval.rational(critical)*inverse_h/(2*(degrees-1))
    return [square(dy)-scale*sums[0], -2*dy*dv-scale*sums[1], square(dv)-scale*sums[2]]


def variance_polynomial(covariance, nuisance):
    constant = covariance[3][3]-2*nuisance*covariance[2][3]+square(nuisance)*covariance[2][2]
    linear = -2*(covariance[1][3]-nuisance*covariance[1][2])
    return [constant, linear, covariance[1][1]]


def variance_constraints(samples, nuisance, critical):
    cutoff_interval = Interval.rational(critical)
    reference = variance_polynomial(samples[0], nuisance)
    inequalities = []
    for cell in (1, 2):
        current = variance_polynomial(samples[cell], nuisance)
        inequalities.append([value-cutoff_interval*base for value, base in zip(current, reference)])
        inequalities.append([base-cutoff_interval*value for value, base in zip(current, reference)])
    return inequalities


def feasible_outer(inequalities):
    domain = [(None, None)]
    for coefficients in inequalities:
        domain = intersection(domain, outer_sublevel(coefficients))
        if not domain:
            break
    return domain


def encode_domain(domain):
    return [[None if value is None else str(value) for value in pair] for pair in domain]


def profile(samples, domain, critical, maximum_boxes=2048):
    if any(lower is None or upper is None for lower, upper in domain):
        return dict(reported=False, reason="unbounded_Fieller_component", evaluated_boxes=0)
    pending = list(domain)
    evaluated, excluded = 0, 0
    while pending and evaluated < maximum_boxes:
        lower, upper = pending.pop()
        nuisance = Interval(Interval.rational(lower).lower, Interval.rational(upper).upper)
        retained = feasible_outer(variance_constraints(samples, nuisance, critical))
        evaluated += 1
        if not retained:
            excluded += 1
            continue
        midpoint = (lower+upper)/2
        point = feasible_outer(variance_constraints(samples, Interval.rational(midpoint), critical))
        if point:
            return dict(reported=False, reason="retained_point_outer_profile", evaluated_boxes=evaluated,
                        excluded_boxes=excluded, nuisance_midpoint=str(midpoint), remaining_coefficients=encode_domain(point))
        if lower == upper:
            return dict(reported=False, reason="numerically_unresolved_point", evaluated_boxes=evaluated)
        pending.extend(((lower, midpoint), (midpoint, upper)))
    return dict(reported=not pending, reason="all_quadratic_branches_excluded" if not pending else "subdivision_budget_retained",
                evaluated_boxes=evaluated, excluded_boxes=excluded, remaining_boxes=len(pending))


def certificate(sample_boxes, rows_per_cell, bank, maximum_boxes=2048):
    if len(sample_boxes) != 4 or rows_per_cell < 4:
        raise ValueError("Four equal-size Gaussian cells with at least four rows are required")
    samples = [symmetric_boxes(matrix) for matrix in sample_boxes]
    ancestry = ancestor_qualification(samples, bank["ratio"])
    if not ancestry["qualified"]:
        return dict(reported=False, reason="ancestry_not_certified", ancestry=ancestry, total_alpha="1/20")
    critical_f = cutoff(bank["fieller"], ALPHA_FIELLER)
    critical_v = cutoff(bank["ratio"], ALPHA_VARIANCE_PAIR)
    if critical_f is None or critical_v is None:
        return dict(reported=False, reason="insufficient_calibration_resolution", ancestry=ancestry, total_alpha="1/20")
    coefficients = fieller_coefficients(samples, rows_per_cell-1, critical_f)
    if coefficients is None or any(not value.finite() for value in coefficients):
        return dict(reported=False, reason="uncertain_Fieller_denominator", ancestry=ancestry, total_alpha="1/20")
    domain = outer_sublevel(coefficients)
    result = profile(samples, domain, critical_v, maximum_boxes)
    return dict(**result, ancestry=ancestry, Fieller_components=encode_domain(domain),
                Fieller_coefficients=[value.pair() for value in coefficients],
                Fieller_cutoff=str(critical_f), variance_ratio_cutoff=str(critical_v), total_alpha="1/20")
