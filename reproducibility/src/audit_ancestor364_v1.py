"""Separate rational arithmetic and Sturm exclusion routines."""
from fractions import Fraction
import sympy as sp
T = sp.Symbol("t")


def point(value):
    value = Fraction(value)
    return value, value


def add(first, second):
    return first[0]+second[0], first[1]+second[1]


def negate(value):
    return -value[1], -value[0]


def subtract(first, second):
    return add(first, negate(second))


def multiply(first, second):
    values = [a*b for a in first for b in second]
    return min(values), max(values)


def divide(first, second):
    assert not second[0] <= 0 <= second[1]
    return multiply(first, (1/second[1], 1/second[0]))


def squared(value):
    upper = max(value[0]*value[0], value[1]*value[1])
    lower = 0 if value[0] <= 0 <= value[1] else min(value[0]*value[0], value[1]*value[1])
    return Fraction(lower), upper


def encoded_covariance(record):
    values = record["sample_covariance_box"]
    return [[(max(Fraction(values[i][j][0]), Fraction(values[j][i][0])),
              min(Fraction(values[i][j][1]), Fraction(values[j][i][1])))
             for j in range(4)] for i in range(4)]


def fieller_polynomial(samples, rows, critical):
    degrees = rows-1
    slopes_y, slopes_v = [], []
    inverse_scatter, q = point(0), [point(0) for _ in range(3)]
    for cell in (0, 3):
        matrix = samples[cell]
        xx, xv, xy = matrix[1][1], matrix[1][2], matrix[1][3]
        slopes_y.append(divide(xy, xx))
        slopes_v.append(divide(xv, xx))
        inverse_scatter = add(inverse_scatter, divide(point(1), multiply(point(degrees), xx)))
        terms = [subtract(matrix[3][3], divide(squared(xy), xx)),
                 add(multiply(point(-2), matrix[2][3]), multiply(point(2), divide(multiply(xy, xv), xx))),
                 subtract(matrix[2][2], divide(squared(xv), xx))]
        q = [add(value, multiply(point(degrees), term)) for value, term in zip(q, terms)]
    dy, dv = subtract(slopes_y[0], slopes_y[1]), subtract(slopes_v[0], slopes_v[1])
    scale = multiply(point(critical/Fraction(2*rows-4)), inverse_scatter)
    return [subtract(squared(dy), multiply(scale, q[0])),
            subtract(multiply(point(-2), multiply(dy, dv)), multiply(scale, q[1])),
            subtract(squared(dv), multiply(scale, q[2]))]


def variance_polynomial(matrix, nuisance):
    constant = add(subtract(matrix[3][3], multiply(point(2), multiply(nuisance, matrix[2][3]))),
                   multiply(squared(nuisance), matrix[2][2]))
    linear = multiply(point(-2), subtract(matrix[1][3], multiply(nuisance, matrix[1][2])))
    return constant, linear, matrix[1][1]


def constraints(samples, nuisance, critical):
    base = variance_polynomial(samples[0], nuisance)
    output = []
    for cell in (1, 2):
        other = variance_polynomial(samples[cell], nuisance)
        output.append([subtract(a, multiply(point(critical), b)) for a, b in zip(other, base)])
        output.append([subtract(b, multiply(point(critical), a)) for a, b in zip(other, base)])
    return output


def polynomial_range(polynomial, lower, upper):
    values = [Fraction(polynomial.eval(sp.Rational(value.numerator, value.denominator))) for value in (lower, upper)]
    if polynomial.degree() == 2:
        a, b, _ = map(Fraction, polynomial.all_coeffs())
        vertex = -b/(2*a)
        if lower <= vertex <= upper:
            values.append(Fraction(polynomial.eval(sp.Rational(vertex.numerator, vertex.denominator))))
    return min(values), max(values)


def no_real_solution(polynomials):
    """Prove emptiness by exact root isolation and signs between all roots."""
    active = []
    for polynomial in polynomials:
        polynomial = sp.Poly(polynomial, T)
        if polynomial.degree() <= 0:
            if polynomial.as_expr() > 0:
                return True
        else:
            active.append(polynomial)
    if not active:
        return False
    product = sp.Poly(1, T)
    for polynomial in active:
        product *= polynomial.clear_denoms(convert=True)[1].primitive()[1]
    roots = product.intervals(eps=sp.Rational(1, 10**32))
    intervals = [(Fraction(pair[0]), Fraction(pair[1])) for pair, multiplicity in roots]
    samples = []
    if intervals:
        samples.append(intervals[0][0]-1)
        samples.extend((first[1]+second[0])/2 for first, second in zip(intervals, intervals[1:]) if first[1] < second[0])
        samples.append(intervals[-1][1]+1)
    else:
        samples.append(Fraction(0))
    for value in samples:
        if all(polynomial.eval(sp.Rational(value.numerator, value.denominator)) <= 0 for polynomial in active):
            return False
    for lower, upper in intervals:
        if not any(polynomial_range(polynomial, lower, upper)[0] > 0 for polynomial in active):
            return False
    return True


def excluded_interval(samples, lower, upper, critical):
    inequalities = constraints(samples, (lower, upper), critical)
    for positive in (True, False):
        polynomials = []
        for constant, linear, quadratic in inequalities:
            c = sp.Rational(constant[0].numerator, constant[0].denominator)
            b = linear[0] if positive else linear[1]
            a = sp.Rational(quadratic[0].numerator, quadratic[0].denominator)
            polynomials.append(a*T*T+sp.Rational(b.numerator, b.denominator)*T+c)
        polynomials.append(-T if positive else T)
        if not no_real_solution(polynomials):
            return False
    return True
