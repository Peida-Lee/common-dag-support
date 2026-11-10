from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import numpy as np
import mpmath as mp
import sympy as sp
import runtime346_v1 as runtime
import audit_ancestor364_v1 as independent
ROOT = Path(__file__).resolve().parent.parent / 'outputs/round365'
ASSETS = Path(__file__).resolve().parent.parent / 'inputs' 


@lru_cache(maxsize=4)
def banks(rows):
    raw = runtime.read(ASSETS / "round364/calibration364_v1" / ("n%d" % rows) / "bank.json")["banks"]
    return {name: {side: list(map(Fraction, values)) for side, values in limits.items()} for name, limits in raw.items()}


def sample_matrices(record):
    return [independent.encoded_covariance(dict(sample_covariance_box=matrix))
            for matrix in record["sample_covariance_boxes"]]


def prove_Fieller_coverage(certificate):
    domain = sp.EmptySet
    for lower, upper in certificate["Fieller_components"]:
        left = -sp.oo if lower is None else sp.Rational(lower)
        right = sp.oo if upper is None else sp.Rational(upper)
        domain = domain.union(sp.Interval(left, right))
    constant, linear, quadratic = [[sp.Rational(value) for value in pair]
                                    for pair in certificate["Fieller_coefficients"]]
    variable = independent.T
    for positive in (True, False):
        polynomial = quadratic[0]*variable**2+linear[0 if positive else 1]*variable+constant[0]
        exact = sp.solve_univariate_inequality(polynomial <= 0, variable, relational=False)
        half = sp.Interval(0, sp.oo) if positive else sp.Interval(-sp.oo, 0)
        difference = exact.intersect(half)-domain
        assert difference == sp.EmptySet, "Fieller outer-set coverage unresolved"


def source_audit(task):
    mp.mp.dps = 100
    record = runtime.read(ROOT / "conditions365_v1" / (task["source"]+".json"))
    model = runtime.read(ROOT / "models365_v1" / (record["model"]+".json"))
    numeric_entries = 0
    with np.load(ROOT / "conditions365_v1" / (task["source"]+".npz")) as original:
        for cell in range(4):
            source = original["cell%d" % cell]
            seed = [365001, record["index"], record["gap_numerator"], record["gap_denominator"],
                    record["rows"], record["replicate"], cell]
            generator = np.random.default_rng(np.random.SeedSequence(seed))
            regenerated = np.zeros((4, 4))
            for i in range(4):
                regenerated[i, i] = generator.chisquare(record["rows"]-1-i)
                regenerated[i, :i] = generator.standard_normal(i)
            assert np.array_equal(source, regenerated)
            rational = [[Fraction(value) for value in row] for row in model["covariances"][cell]]
            covariance = mp.matrix([[mp.mpf(value.numerator)/value.denominator for value in row] for row in rational])
            lower = mp.cholesky(covariance)
            bartlett = mp.matrix(4)
            for i in range(4):
                for j in range(i+1):
                    fraction = Fraction.from_float(float(source[i, j]))
                    value = mp.mpf(fraction.numerator)/fraction.denominator
                    bartlett[i, j] = mp.sqrt(value) if i == j else value
            product = lower*bartlett
            scatter = product*product.T/(record["rows"]-1)
            for i in range(4):
                for j in range(4):
                    lo, hi = map(Fraction, record["sample_covariance_boxes"][cell][i][j])
                    assert mp.mpf(lo.numerator)/lo.denominator <= scatter[i, j]
                    assert scatter[i, j] <= mp.mpf(hi.numerator)/hi.denominator
                    numeric_entries += 1
    samples, references = sample_matrices(record), banks(record["rows"])
    certificate = record["certificate"]
    raw_p_checks = 0
    for decision in certificate["ancestry"]["decisions"]:
        target, coordinate = decision["target"], decision["coordinate"]
        ratio = independent.divide(samples[target+1][coordinate][coordinate], samples[0][coordinate][coordinate])
        reciprocal = independent.divide(independent.point(1), ratio)
        lo, hi = max(Fraction(1), ratio[0], reciprocal[0]), max(ratio[1], reciprocal[1])
        p_lower = Fraction(1+sum(value >= hi for value in references["ratio"]["lower"]), 1024)
        p_upper = Fraction(1+sum(value >= lo for value in references["ratio"]["upper"]), 1024)
        saved_lower, saved_upper = map(Fraction, decision["raw_P_interval"])
        assert saved_lower <= p_lower <= p_upper <= saved_upper
        assert (decision["status"] == "certified") == (saved_upper <= Fraction(1, 300))
        raw_p_checks += 1
    if "Fieller_coefficients" in certificate:
        expected = independent.fieller_polynomial(samples, record["rows"], Fraction(certificate["Fieller_cutoff"]))
        for bounds, saved in zip(expected, certificate["Fieller_coefficients"]):
            assert Fraction(saved[0]) <= bounds[0] <= bounds[1] <= Fraction(saved[1])
    if certificate["reported"]:
        prove_Fieller_coverage(certificate)
    assert not record["one_box_reported"] or certificate["reported"]
    return dict(status="completed", source_conditions=1, covariance_entries=numeric_entries,
                raw_P_intervals=raw_p_checks, Fieller_sets_proved=int(certificate["reported"]),
                reported=certificate["reported"], false_report=certificate["reported"] and model["gap"] == "0")


def profile_audit(task):
    record = runtime.read(ROOT / "conditions365_v1" / (task["source"]+".json"))
    samples = sample_matrices(record)
    critical = Fraction(record["certificate"]["variance_ratio_cutoff"])
    pending = [(Fraction(task["lower"]), Fraction(task["upper"]))]
    evaluated, leaves = 0, 0
    while pending:
        lower, upper = pending.pop()
        evaluated += 1
        assert evaluated <= 4096, "Independent profile exclusion budget exhausted"
        if independent.excluded_interval(samples, lower, upper, critical):
            leaves += 1
        else:
            midpoint = (lower+upper)/2
            assert lower < midpoint < upper
            pending.extend(((lower, midpoint), (midpoint, upper)))
    return dict(status="completed", excluded_profile_parts=1, rational_subregions=leaves, Sturm_evaluations=evaluated)


def operate(task):
    return source_audit(task) if task["kind"] == "source" else profile_audit(task)
