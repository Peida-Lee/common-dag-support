"""Outward scalar arithmetic and simultaneous unknown-scale covariance boxes."""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("Invalid closed interval")

    @classmethod
    def point(cls, value):
        return cls(float(value), float(value))

    @classmethod
    def rational(cls, value):
        value = Fraction(value)
        rounded = float(value)
        represented = Fraction.from_float(rounded)
        lower = math.nextafter(rounded, -math.inf) if represented > value else rounded
        upper = math.nextafter(rounded, math.inf) if represented < value else rounded
        return cls(lower, upper)

    def contains(self, value):
        return self.lower <= value <= self.upper

    def excludes_zero(self):
        return self.upper < 0 or self.lower > 0

    def finite(self):
        return math.isfinite(self.lower) and math.isfinite(self.upper)

    def pair(self):
        return [value if math.isfinite(value) else ("+inf" if value > 0 else "-inf")
                for value in (self.lower, self.upper)]

    def __neg__(self):
        return Interval(-self.upper, -self.lower)

    def __add__(self, other):
        other = as_interval(other)
        lower, upper = self.lower+other.lower, self.upper+other.upper
        if math.isnan(lower) or math.isnan(upper):
            return universal()
        return Interval(math.nextafter(lower, -math.inf), math.nextafter(upper, math.inf))

    __radd__ = __add__

    def __sub__(self, other):
        return self + -as_interval(other)

    def __rsub__(self, other):
        return as_interval(other) + -self

    def __mul__(self, other):
        other = as_interval(other)
        products = [0. if first == 0 or second == 0 else first*second
                    for first in (self.lower, self.upper) for second in (other.lower, other.upper)]
        if self.lower == self.upper == 0 or other.lower == other.upper == 0:
            return Interval.point(0)
        return Interval(math.nextafter(min(products), -math.inf), math.nextafter(max(products), math.inf))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_interval(other)
        if other.contains(0):
            return universal()
        reciprocal = Interval(math.nextafter(1/other.upper, -math.inf),
                              math.nextafter(1/other.lower, math.inf))
        return self * reciprocal

    def sqrt(self):
        if self.upper < 0:
            raise ValueError("A negative interval has no real square root")
        lower = sqrt_endpoint(max(0., self.lower), upper=False)
        upper = sqrt_endpoint(self.upper, upper=True)
        return Interval(lower, upper)


def as_interval(value):
    return value if isinstance(value, Interval) else Interval.point(value)


def universal():
    return Interval(-math.inf, math.inf)


def sqrt_endpoint(value, upper):
    if value == 0 or value == math.inf:
        return value
    root = math.sqrt(value)
    exact = Fraction.from_float(value)
    square = Fraction.from_float(root)**2
    if upper and square < exact:
        root = math.nextafter(root, math.inf)
    if not upper and square > exact:
        root = math.nextafter(root, -math.inf)
    result = Fraction.from_float(root)**2
    if (upper and result < exact) or (not upper and result > exact):
        raise ArithmeticError("The scalar square-root enclosure requires additional precision")
    return root


def tail_parameters(degrees, dimension, cells, alpha):
    alpha = Fraction(alpha)
    if degrees < 1 or dimension < 1 or cells < 1 or not 0 < alpha < 1:
        raise ValueError("Positive degrees, dimensions, cells and a probability budget are required")
    exponent = 0
    while (1 << exponent)*alpha < 2*cells*dimension**2:
        exponent += 1
    bits = 64
    integer_root = math.isqrt(degrees*exponent*(1 << (2*bits)))
    upper_root = Fraction(integer_root+1, 1 << bits)
    lower_pivot = Fraction(degrees)-2*upper_root
    upper_pivot = Fraction(degrees)+2*upper_root+2*exponent
    return dict(exponent=exponent, lower_pivot=lower_pivot, upper_pivot=upper_pivot,
                alpha=alpha, forms=cells*dimension**2)


def sample_covariance_box(values):
    """Enclose exact sample covariances of the finite supplied input values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two complete finite rows are required")
    count, dimension = values.shape
    augmented = np.column_stack((np.ones(count), values))
    gram = augmented.T @ augmented
    absolute = np.abs(augmented)
    absolute_gram = absolute.T @ absolute
    if not np.isfinite(gram).all() or not np.isfinite(absolute_gram).all():
        raise ArithmeticError("The augmented Gram calculation overflowed")
    count_roundoff = Interval.rational(Fraction(count, 1 << 53))
    if count_roundoff.upper >= .5:
        raise ArithmeticError("The declared dot-product error bound is unavailable")
    gamma = (count_roundoff/(1-count_roundoff)).upper
    denominator = Interval.point(1)-gamma
    minimum_positive = math.ulp(0.)
    underflow = ((Interval.point(2*count)*minimum_positive)/denominator).upper
    maximum_radius = 0.
    gram_boxes = []
    for row in range(dimension+1):
        bounds = []
        for column in range(dimension+1):
            absolute_upper = ((Interval.point(absolute_gram[row, column])+underflow)/denominator).upper
            radius = (Interval.point(gamma)*absolute_upper+underflow).upper
            maximum_radius = max(maximum_radius, radius)
            center = Interval.point(gram[row, column])
            bounds.append(Interval((center-radius).lower, (center+radius).upper))
        gram_boxes.append(bounds)
    covariance = []
    for row in range(dimension):
        covariance.append([(gram_boxes[row+1][column+1]-
                            gram_boxes[0][row+1]*gram_boxes[0][column+1]/count)/(count-1)
                           for column in range(dimension)])
    return covariance, dict(rows=count, dimension=dimension, degrees=count-1,
                            dot_gamma_upper=gamma, underflow_allowance=underflow,
                            maximum_Gram_radius=maximum_radius,
                            numerical_model="IEEE_dot_product_roundoff_with_absolute_underflow_allowance")


def variance_region(sample_variance, degrees, parameters):
    lower = max(0., sample_variance.lower)
    upper = max(0., sample_variance.upper)
    if sample_variance.upper < 0:
        raise ArithmeticError("The enclosed sample variance is strictly negative")
    lower_endpoint = ((Interval.point(degrees)*lower)/Interval.rational(parameters["upper_pivot"])).lower
    upper_endpoint = math.inf
    if parameters["lower_pivot"] > 0:
        upper_endpoint = ((Interval.point(degrees)*upper)/Interval.rational(parameters["lower_pivot"])).upper
    return Interval(max(0., lower_endpoint), upper_endpoint)


def covariance_region(values, cells, alpha=Fraction(1, 20)):
    sample, numerical = sample_covariance_box(values)
    dimension = len(sample)
    degrees = len(values)-1
    parameters = tail_parameters(degrees, dimension, cells, alpha)
    region = [[None]*dimension for _ in range(dimension)]
    for row in range(dimension):
        region[row][row] = variance_region(sample[row][row], degrees, parameters)
        for column in range(row):
            plus = sample[row][row]+sample[column][column]+2*sample[row][column]
            minus = sample[row][row]+sample[column][column]-2*sample[row][column]
            plus_region = variance_region(plus, degrees, parameters)
            minus_region = variance_region(minus, degrees, parameters)
            interval = (plus_region-minus_region)/4
            region[row][column] = region[column][row] = interval
    metadata = dict(**numerical, tail_exponent=parameters["exponent"],
                    simultaneous_alpha=str(parameters["alpha"]), total_forms=parameters["forms"],
                    positive_lower_pivot=parameters["lower_pivot"] > 0)
    return region, sample, metadata


def subtract_boxes(first, second):
    return [[a-b for a, b in zip(row, other)] for row, other in zip(first, second)]


def encode_box(box):
    return [[entry.pair() for entry in row] for row in box]


def decode_box(box):
    return [[Interval(*(float(value) for value in entry)) for entry in row] for row in box]
