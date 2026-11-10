"""Joint polynomial reference on the same final conditions."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import time
import random
import z3
import runtime346_v1 as runtime
import audit_ancestor364_v1 as independent
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent
SOLVER_MILLISECONDS = 2000
STURM_PARTITION_LIMIT = 16384


def covariance_boxes(record):
    return [independent.encoded_covariance(dict(sample_covariance_box=matrix))
            for matrix in record["sample_covariance_boxes"]]


def polynomial_coefficients(samples, critical):
    coefficients = []
    for cell in (1, 2):
        for first, second in ((samples[cell], samples[0]), (samples[0], samples[cell])):
            matrix = [[independent.subtract(first[i][j], independent.multiply(independent.point(critical), second[i][j]))
                       for j in range(4)] for i in range(4)]
            coefficients.append([matrix[3][3], independent.multiply(independent.point(-2), matrix[1][3]),
                                 independent.multiply(independent.point(-2), matrix[2][3]), matrix[1][1],
                                 independent.multiply(independent.point(2), matrix[1][2]), matrix[2][2]])
    return coefficients


def monomials(t, b):
    return (1, t, b, t * t, t * b, b * b)


def lower_value(coefficients, t, b):
    return sum(bounds[0 if term >= 0 else 1] * term for bounds, term in zip(coefficients, monomials(t, b)))


def rational(value):
    value = Fraction(value)
    return z3.RealVal(value.numerator) / z3.RealVal(value.denominator)


def joint_formula(coefficients, components, t, b):
    domains = []
    for lower, upper in components:
        bounds = []
        if lower is not None:
            bounds.append(b >= rational(lower))
        if upper is not None:
            bounds.append(b <= rational(upper))
        domains.append(z3.And(*bounds))
    orthants = []
    for sign_t, sign_b in product((-1, 1), repeat=2):
        signs = (1, sign_t, sign_b, 1, sign_t * sign_b, 1)
        expressions = [sum(rational(bounds[0 if sign >= 0 else 1]) * term
                           for bounds, sign, term in zip(polynomial, signs, monomials(t, b)))
                       for polynomial in coefficients]
        orthants.append(z3.And(sign_t * t >= 0, sign_b * b >= 0, *[expression <= 0 for expression in expressions]))
    return z3.And(z3.Or(*domains), z3.Or(*orthants))


def accepted_point(coefficients, components, t, b):
    in_domain = any((lower is None or Fraction(lower) <= b) and (upper is None or b <= Fraction(upper))
                    for lower, upper in components)
    return in_domain and all(lower_value(polynomial, t, b) <= 0 for polynomial in coefficients)


def exact_model_fraction(value):
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    approximation = value.approx(128)
    return Fraction(approximation.numerator_as_long(), approximation.denominator_as_long())


def rational_witness(model, t_variable, b_variable, coefficients, components):
    t = exact_model_fraction(model.eval(t_variable, model_completion=True))
    b = exact_model_fraction(model.eval(b_variable, model_completion=True))
    if accepted_point(coefficients, components, t, b):
        return [str(t), str(b)]
    for precision in (32, 48, 64, 96, 128, 192):
        step = Fraction(1, 2 ** precision)
        for offset_t, offset_b in product((-1, 0, 1), repeat=2):
            candidate_t, candidate_b = t + offset_t * step, b + offset_b * step
            if accepted_point(coefficients, components, candidate_t, candidate_b):
                return [str(candidate_t), str(candidate_b)]
    return None


def certify_exclusion(samples, components, critical):
    if any(lower is None or upper is None for lower, upper in components):
        return dict(status="unbounded_component_unresolved", evaluations=0)
    pending = [(Fraction(lower), Fraction(upper)) for lower, upper in components]
    evaluated = 0
    while pending and evaluated < STURM_PARTITION_LIMIT:
        lower, upper = pending.pop()
        evaluated += 1
        if independent.excluded_interval(samples, lower, upper, critical):
            continue
        midpoint = (lower + upper) / 2
        if not lower < midpoint < upper:
            return dict(status="unresolved_point", evaluations=evaluated)
        pending.extend(((lower, midpoint), (midpoint, upper)))
    return dict(status="passed" if not pending else "partition_budget_unresolved", evaluations=evaluated)


def condition(task):
    source = runtime.read(ASSETS / "round365/conditions365_v1" / (task["source"] + ".json"))
    certificate = source["certificate"]
    output = dict(status="completed", source=task["source"], rows=source["rows"], index=source["index"],
                  gap=str(Fraction(source["gap_numerator"], source["gap_denominator"])),
                  original_report=certificate["reported"], original_reason=certificate["reason"],
                  solver_version=z3.get_full_version(), solver_timeout_ms=SOLVER_MILLISECONDS,
                  reported=False, decision="unqualified", certification="not_applicable")
    if not certificate["ancestry"]["qualified"]:
        runtime.save(ROOT / "conditions369_v2" / (task["source"] + ".json"), output)
        return output
    samples = covariance_boxes(source)
    critical = Fraction(certificate["variance_ratio_cutoff"])
    coefficients = polynomial_coefficients(samples, critical)
    fieller = [tuple(map(Fraction, bounds)) for bounds in certificate["Fieller_coefficients"]]
    zero = independent.point(0)
    coefficients.append([fieller[0], zero, fieller[1], zero, zero, fieller[2]])
    components = certificate["Fieller_components"]
    t, b = z3.Reals("t b")
    formula = joint_formula(coefficients, components, t, b)
    solver = z3.SolverFor("QF_NRA")
    solver.set(timeout=SOLVER_MILLISECONDS)
    solver.add(formula)
    began = time.monotonic()
    decision = solver.check()
    output.update(decision=str(decision), solver_seconds=time.monotonic() - began,
                  coefficients=[[[str(value) for value in bounds] for bounds in polynomial] for polynomial in coefficients],
                  Fieller_components=components)
    if decision == z3.sat:
        model = solver.model()
        witness = rational_witness(model, t, b, coefficients, components)
        output.update(model_t=model.eval(t, model_completion=True).sexpr(),
                      model_b=model.eval(b, model_completion=True).sexpr(), rational_witness=witness,
                      certification="rational_witness_verified" if witness is not None else "algebraic_solver_only")
    elif decision == z3.unsat:
        proof = certify_exclusion(samples, components, critical)
        output.update(certification=proof["status"], Sturm_evaluations=proof["evaluations"], reported=proof["status"] == "passed")
    else:
        output.update(certification="solver_unresolved", solver_reason=solver.reason_unknown())
    runtime.save(ROOT / "conditions369_v2" / (task["source"] + ".json"), output)
    if decision == z3.unknown:
        (ROOT / "conditions369_v2" / (task["source"] + ".smt2")).write_text(solver.to_smt2())
    return {key:value for key,value in output.items() if key not in ("coefficients", "Fieller_components", "model_t", "model_b")}
