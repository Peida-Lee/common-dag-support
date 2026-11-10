"""Final optical parsing, source contrasts and group F inference."""
from pathlib import Path
import re
import numpy as np
from scipy import stats
EXPERIMENTS = (69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82)


def parse_values(values, experiment):
    values = np.asarray(values, dtype=float)
    expected = 10 if experiment == 81 else 11
    if values.ndim != 2 or values.shape[1] != expected or not np.isfinite(values).all():
        raise ValueError("Optical source has incompatible columns or nonfinite values")
    pixels = values[:, :9].copy()
    response = values[:, -1].copy()
    source = np.zeros(len(values)) if experiment == 81 else values[:, 9].copy()
    return pixels, source, response


def catalogue(folder):
    files = {}
    for path in Path(folder).glob("*.txt"):
        match = re.search(r"exp_no_(\d+)_", path.name)
        if match:
            identifier = int(match.group(1))
            if identifier in files:
                raise ValueError("Duplicate experiment identifier")
            files[identifier] = path
    if tuple(sorted(files)) != EXPERIMENTS:
        raise ValueError("The complete frozen optical catalogue is required")
    return files


def source_contrasts(values, source):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) != len(source):
        raise ValueError("Contrast values and source rows disagree")
    contrasts, mapping = [], []
    for level in np.unique(source):
        indices = np.flatnonzero(source == level)
        selected = values[indices]
        count = len(indices)
        if count > 1:
            k = np.arange(1, count, dtype=float)[:, None]
            contrasts.append(np.sqrt(k / (k + 1)) *
                             (selected[1:] - np.cumsum(selected, axis=0)[:-1] / k))
        mapping.append(dict(level=float(level), original_rows=indices.tolist(), contrasts=max(0, count-1)))
    result = np.concatenate(contrasts) if contrasts else np.empty((0, values.shape[1]))
    assert len(result) == len(values) - len(mapping)
    return result, mapping


def rank(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
    return int(np.count_nonzero(singular > tolerance))


def group_tests(design, response, groups, alpha=.05):
    design = np.column_stack((np.ones(len(response)), design))
    full_rank = rank(design)
    degrees = len(response) - full_rank
    coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ coefficient
    rss = float(residual @ residual)
    threshold = alpha / len(groups)
    records = []
    for pixel, group in enumerate(groups):
        removed = {column + 1 for column in group}
        reduced = design[:, [column for column in range(design.shape[1]) if column not in removed]]
        reduced_rank = rank(reduced)
        record = dict(pixel=pixel, feature_columns=group, numerator_df=full_rank-reduced_rank,
                      denominator_df=degrees, original_P=None, statistic=None, rejected=False,
                      unresolved=True, threshold=threshold)
        if degrees > 0 and full_rank - reduced_rank == len(group) and rss > 0:
            reduced_coefficient = np.linalg.lstsq(reduced, response, rcond=None)[0]
            reduced_residual = response - reduced @ reduced_coefficient
            reduced_rss = float(reduced_residual @ reduced_residual)
            difference = reduced_rss - rss
            tolerance = 100 * np.finfo(float).eps * max(1., float(response @ response))
            if difference < -tolerance:
                raise ArithmeticError("Reduced regression has materially smaller RSS")
            statistic = max(0., difference) / len(group) / (rss / degrees)
            probability = float(stats.f.sf(statistic, len(group), degrees))
            unresolved = not np.isfinite(probability) or abs(probability-threshold) <= 1e-10
            record.update(original_P=probability, statistic=statistic, reduced_RSS=reduced_rss,
                          unresolved=unresolved, rejected=not unresolved and probability < threshold)
        records.append(record)
    normal = stats.normaltest(residual) if len(residual) >= 8 else None
    diagnostics = dict(residual_skew=float(stats.skew(residual)),
                       residual_excess_kurtosis=float(stats.kurtosis(residual)),
                       residual_lag1=float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
                       transformed_normality_original_P=None if normal is None else float(normal.pvalue),
                       diagnostic_role="working_model_checks_on_projected_regression_residuals")
    singular = np.linalg.svd(design, compute_uv=False)
    return dict(status="completed", rows=len(response), columns=design.shape[1], rank=full_rank,
                denominator_df=degrees, full_RSS=rss, coefficients=coefficient.tolist(),
                singular_values=singular.tolist(), groups=records, diagnostics=diagnostics,
                detected_pixels=[item["pixel"] for item in records if item["rejected"]],
                unresolved_pixels=[item["pixel"] for item in records if item["unresolved"]],
                alpha=alpha, family="nine_pixel_groups_per_acquisition_and_basis", truth_read=False)


def pixel_projection(feature_edges, groups, response_column):
    selected = {int(parent) for parent, child in feature_edges if int(child) == response_column}
    return [pixel for pixel, group in enumerate(groups) if selected.intersection(group)]
