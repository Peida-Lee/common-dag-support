"""Parameterized four-node population model."""
from fractions import Fraction
import response_factors356_v1 as factors


def encoded(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded(item) for item in value]
    return value


def family(index, positive):
    a = Fraction(4+index % 3, 8)
    c = Fraction(3+(index//3) % 3, 8)
    b = Fraction(4+index % 3, 4)
    k = c+Fraction(2+index % 2, 8)
    l = Fraction(1+(index//9) % 2, 4)
    e = -Fraction(3+index % 4, 4)
    s = Fraction(1, 4) if positive else Fraction(0)
    coefficients = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    coefficients[1][0], coefficients[2][1], coefficients[3][2], coefficients[3][0] = a, c, b, s
    response = factors.response_from_lower(coefficients)
    alternate = [(Fraction(1), a, a*c, e*a*c), (Fraction(0), Fraction(1), k, e*k),
                 (Fraction(0), l, Fraction(1), e)]
    candidates = [[tuple(response[i][j] for i in range(4)), alternate[j]] for j in range(3)]+[None]
    contrasts, covariances, scales = [], [factors.identity(4)], []
    for target in range(3):
        raw = factors.contrast(*candidates[target])
        scale = Fraction(3, 4)/max(sum(abs(value) for value in row) for row in raw)
        contrast = [[scale*value for value in row] for row in raw]
        assert set(factors.factor_candidates(contrast, target)) == set(candidates[target])
        contrasts.append(contrast)
        covariances.append([[Fraction(i == j)+contrast[i][j] for j in range(4)] for i in range(4)])
        scales.append(scale)
    return dict(index=index, positive=positive, parameters=dict(a=a, c=c, b=b, k=k, l=l, e=e, s=s),
                physical=coefficients, response=response, candidates=candidates,
                contrasts=contrasts, covariances=covariances, scales=scales)
