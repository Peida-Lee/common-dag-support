from pathlib import Path
from fractions import Fraction
import runtime346_v1 as runtime
import audit_ancestor364_v1 as sturm
ROOT = Path(__file__).resolve().parent.parent / 'outputs/joint'
ASSETS = Path(__file__).resolve().parent.parent / 'outputs' 


def audit(task):
    source = runtime.read(ASSETS / "round365/conditions365_v1" / (task["identifier"] + ".json"))
    result = runtime.read(ROOT / "conditions369_v2" / (task["identifier"] + ".json"))
    assert result["original_report"] == source["certificate"]["reported"]
    if result["decision"] == "unqualified":
        assert not source["certificate"]["ancestry"]["qualified"] and not result["reported"]
        return dict(status="completed", source_conditions=1, unqualified=1)
    assert source["certificate"]["ancestry"]["qualified"]
    matrices = [sturm.encoded_covariance(dict(sample_covariance_box=matrix))
                for matrix in source["sample_covariance_boxes"]]
    critical = Fraction(source["certificate"]["variance_ratio_cutoff"])
    pairs = ((1, 0), (0, 1), (2, 0), (0, 2))
    coordinates = ((3, 3, 1), (1, 3, -2), (2, 3, -2), (1, 1, 1), (1, 2, 2), (2, 2, 1))
    recomputed = []
    for first, second in pairs:
        polynomial = []
        for i, j, scale in coordinates:
            lo = matrices[first][i][j][0] - critical * matrices[second][i][j][1]
            hi = matrices[first][i][j][1] - critical * matrices[second][i][j][0]
            polynomial.append((min(scale * lo, scale * hi), max(scale * lo, scale * hi)))
        recomputed.append(polynomial)
    fieller = [tuple(map(Fraction, bounds)) for bounds in source["certificate"]["Fieller_coefficients"]]
    zero = (Fraction(0), Fraction(0))
    recomputed.append([fieller[0], zero, fieller[1], zero, zero, fieller[2]])
    saved = [[tuple(map(Fraction, bounds)) for bounds in polynomial] for polynomial in result["coefficients"]]
    assert saved == recomputed
    components = source["certificate"]["Fieller_components"]
    assert components == result["Fieller_components"]
    if result["decision"] == "sat":
        assert result["certification"] == "rational_witness_verified" and not result["reported"]
        t, b = map(Fraction, result["rational_witness"])
        assert any((lo is None or Fraction(lo) <= b) and (hi is None or b <= Fraction(hi)) for lo, hi in components)
        terms = (Fraction(1), t, b, t*t, t*b, b*b)
        for polynomial in recomputed:
            lower = sum(min(lo * term, hi * term) for (lo, hi), term in zip(polynomial, terms))
            assert lower <= 0
        return dict(status="completed", source_conditions=1, coefficient_intervals=30,
                    rational_witnesses=1, rational_point_inequalities=5)
    if result["decision"] == "unsat":
        assert result["certification"] == "passed" and result["reported"]
        assert all(lo is not None and hi is not None for lo, hi in components)
        pending = [(Fraction(lo), Fraction(hi)) for lo, hi in components]
        evaluations = 0
        while pending:
            lo, hi = pending.pop()
            evaluations += 1
            assert evaluations <= 16384
            if not sturm.excluded_interval(matrices, lo, hi, critical):
                middle = (lo + hi) / 2
                assert lo < middle < hi
                pending.extend(((lo, middle), (middle, hi)))
        return dict(status="completed", source_conditions=1, coefficient_intervals=30,
                    excluded_joint_sets=1, Sturm_evaluations=evaluations,
                    false_reports=int(source["gap_numerator"] == 0))
    return dict(status="completed", source_conditions=1, coefficient_intervals=30,
                unresolved=1, reported=result["reported"])
