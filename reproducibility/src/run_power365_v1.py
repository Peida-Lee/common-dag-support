"""Final independent-Gaussian power conditions."""
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import numpy as np
import runtime346_v1 as runtime
from Gaussian_sources365_v1 import model, encoded, source_draw
from ancestor_profile364_v1 import (certificate, cutoff, ALPHA_VARIANCE_PAIR, symmetric_boxes,
                                   feasible_outer, variance_constraints)
from outward_covariance359_v1 import Interval, tail_parameters, variance_region, subtract_boxes
from common_DAG_regions359_v1 import factor_regions, missing_query
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent
BUDGETS = (8192, 32768, 131072, 524288)
GAPS = (Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
INDICES = (0, 1, 3, 4)


@lru_cache(maxsize=4)
def bank(count):
    return runtime.read(ASSETS / "round364/calibration364_v1" / ("n%d" % count) / "bank.json")["banks"]


def population_region(sample, rows, alpha):
    parameters = tail_parameters(rows-1, len(sample), 4, alpha)
    output = [[None]*len(sample) for _ in sample]
    for i in range(len(sample)):
        output[i][i] = variance_region(sample[i][i], rows-1, parameters)
        for j in range(i):
            plus = variance_region(sample[i][i]+sample[j][j]+2*sample[i][j], rows-1, parameters)
            minus = variance_region(sample[i][i]+sample[j][j]-2*sample[i][j], rows-1, parameters)
            output[i][j] = output[j][i] = (plus-minus)/4
    return output


def interval_query(samples, rows, order_qualified):
    regions = [population_region(sample, rows, Fraction(1, 25) if order_qualified else Fraction(1, 20))
               for sample in samples]
    domains = [None]*4
    for target in range(3):
        domains[target], _ = factor_regions(subtract_boxes(regions[target+1], regions[0]), target)
        if domains[target] is None:
            return dict(reported=False, reason="factor_qualification_failure")
        if order_qualified:
            retained = []
            for column in domains[target]:
                if all(column[earlier].contains(0) for earlier in range(target)):
                    column = list(column)
                    for earlier in range(target):
                        column[earlier] = Interval.point(0)
                    retained.append(column)
            domains[target] = retained
            if not retained:
                return dict(reported=False, reason="order_region_qualification_failure")
    return missing_query(domains, 3, (1, 2), (1, 2), ())


def one_box_result(samples, result, references):
    if not result["ancestry"]["qualified"]:
        return False
    components = result.get("Fieller_components")
    if components is None:
        return False
    critical = cutoff(references["ratio"], ALPHA_VARIANCE_PAIR)
    for lower, upper in components:
        if lower is None or upper is None:
            return False
        nuisance = Interval(Interval.rational(Fraction(lower)).lower, Interval.rational(Fraction(upper)).upper)
        if feasible_outer(variance_constraints(samples, nuisance, critical)):
            return False
    return True


def condition(task):
    name = task["model"]
    source = runtime.read(ROOT / "models365_v1" / (name+".json"))
    sample_boxes, primitive_values = [], {}
    for cell in range(4):
        covariance = [[Fraction(value) for value in row] for row in source["covariances"][cell]]
        seed = [365001, task["index"], task["gap_numerator"], task["gap_denominator"],
                task["rows"], task["replicate"], cell]
        generator = np.random.default_rng(np.random.SeedSequence(seed))
        boxes, primitive = source_draw(covariance, task["rows"], generator)
        sample_boxes.append(boxes)
        primitive_values["cell%d" % cell] = primitive
    references = bank(task["rows"])
    result = certificate(sample_boxes, task["rows"], references)
    samples = [symmetric_boxes(values) for values in sample_boxes]
    one_box = one_box_result(samples, result, references)
    assert not one_box or result["reported"]
    original = interval_query(samples, task["rows"], False)
    ordered = (interval_query(samples, task["rows"], True) if result["ancestry"]["qualified"]
               else dict(reported=False, reason="ancestry_not_certified"))
    folder = ROOT / "conditions365_v1"
    np.savez_compressed(folder / (task["identifier"]+".npz"), **primitive_values)
    runtime.save(folder / (task["identifier"]+".json"),
                 dict(status="completed", **task, sample_covariance_boxes=sample_boxes, certificate=result,
                      one_box_reported=one_box, original_interval=original, order_qualified_interval=ordered,
                      source_type="Gaussian_centered_scatter_Bartlett", source_seed_rule=365001))
    return dict(status="completed", model=name, index=task["index"], gap=source["gap"], rows=task["rows"],
                replicate=task["replicate"], reported=result["reported"], qualified=result["ancestry"]["qualified"],
                one_box_reported=one_box, original_interval_reported=original["reported"],
                ordered_interval_reported=ordered["reported"], reason=result["reason"],
                evaluated_boxes=result.get("evaluated_boxes", 0))
