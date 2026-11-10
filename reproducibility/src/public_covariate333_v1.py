"""Target-blind polynomial nuisance projection."""
import numpy as np


def polynomial_adjust(controls, outcome):
    controls = np.asarray(controls, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    assert controls.ndim == 2 and len(controls) == len(outcome)
    lower, upper = controls.min(0), controls.max(0)
    scaled = np.divide(2 * (controls - lower), upper - lower,
                       out=np.zeros_like(controls), where=upper > lower)
    scaled[:, upper > lower] -= 1
    columns = [np.ones(len(outcome))]
    columns.extend(scaled[:, index] ** degree
                   for index in range(controls.shape[1]) for degree in (1, 2, 3))
    columns.extend(scaled[:, left] * scaled[:, right]
                   for left in range(controls.shape[1])
                   for right in range(left + 1, controls.shape[1]))
    design = np.column_stack(columns)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ coefficients
    assert np.isfinite(residuals).all()
    return residuals, dict(
        columns=design.shape[1], rank=int(rank), coefficients=coefficients.tolist(),
        singular_values=singular_values.tolist(), lower=lower.tolist(), upper=upper.tolist(),
        target_and_assignment_read=False, basis="intercept_cubic_marginals_pairwise_linear_products",
    )
