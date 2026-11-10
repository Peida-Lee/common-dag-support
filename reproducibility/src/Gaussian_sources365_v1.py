"""Rational non-unit covariance families and enclosed Bartlett scatters."""

from pathlib import Path
import sys

ASSETS = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ASSETS / folder) for folder in ("round356", "round359", "round362")]
import response_factors356_v1 as factors
from validate_complete_support362_v1 import family, encoded
from outward_covariance359_v1 import Interval, encode_box

from fractions import Fraction
import numpy as np


def model(index, gap):
    original = family(index, bool(gap))
    coefficients = [row[:] for row in original["physical"]]
    coefficients[3][0] = gap
    response = factors.response_from_lower(coefficients)
    candidates = [[tuple(response[i][j] for i in range(4)), original["candidates"][j][1]] for j in range(3)]+[None]
    baseline = [[Fraction(i == j)+Fraction((-1)**(index+i+j), 128 if i == j else 96)
                 for j in range(4)] for i in range(4)]
    perturbation = [[baseline[i][j]-int(i == j) for j in range(4)] for i in range(4)]
    assert max(sum(abs(value) for value in row) for row in perturbation) < Fraction(1, 8)
    covariances, scales = [baseline], []
    for target in range(3):
        raw = factors.contrast(*candidates[target])
        scale = Fraction(3, 4)/max(sum(abs(value) for value in row) for row in raw)
        covariances.append([[baseline[i][j]+scale*raw[i][j] for j in range(4)] for i in range(4)])
        scales.append(scale)
    parameters = dict(original["parameters"], s=gap)
    return dict(index=index, gap=gap, parameters=parameters, physical=coefficients, response=response,
                candidates=candidates, covariances=covariances, scales=scales,
                covariance_eigenvalue_bounds=[Fraction(1, 8), Fraction(15, 8)])


def cholesky_box(covariance):
    size = len(covariance)
    lower = [[Interval.point(0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i+1):
            value = Interval.rational(covariance[i][j])
            for k in range(j):
                value = value-lower[i][k]*lower[j][k]
            if i == j:
                if value.lower <= 0:
                    raise ArithmeticError("The Cholesky pivot cannot be enclosed positively")
                lower[i][j] = value.sqrt()
            else:
                lower[i][j] = value/lower[j][j]
    return lower


def source_draw(covariance, rows, generator):
    size, degrees = len(covariance), rows-1
    if degrees < size:
        raise ValueError("This positive-definite Bartlett source needs n-1 at least p")
    primitive = np.zeros((size, size))
    bartlett = [[Interval.point(0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        primitive[i, i] = generator.chisquare(degrees-i)
        bartlett[i][i] = Interval.point(primitive[i, i]).sqrt()
        primitive[i, :i] = generator.standard_normal(i)
        for j in range(i):
            bartlett[i][j] = Interval.point(primitive[i, j])
    lower = cholesky_box(covariance)
    product = [[sum((lower[i][k]*bartlett[k][j] for k in range(size)), Interval.point(0))
                for j in range(size)] for i in range(size)]
    sample = [[sum((product[i][k]*product[j][k] for k in range(size)), Interval.point(0))/degrees
               for j in range(size)] for i in range(size)]
    return encode_box(sample), primitive
