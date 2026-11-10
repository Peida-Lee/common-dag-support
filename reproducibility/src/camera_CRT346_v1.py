"""Source-faithful command generation and target-blind trajectory statistics."""

from itertools import combinations_with_replacement

import numpy as np
from numba import njit


ROOTS = ("red", "green", "blue", "pol_1", "pol_2")
OUTPUTS = ("current", "angle_1", "angle_2", "ir_1", "ir_2", "ir_3",
           "vis_1", "vis_2", "vis_3", "im")
CONDITIONS = ("ar_1_uniform_ref", "ar_1_uniform_pol_1_90", "ar_1_uniform_pol_12_30")
PHI = (0.5, 0.6, 0.7, 0.8, 0.9)
MEANS = ((128., 128., 128., 0., 0.),
         (128., 128., 128., 0., 90.),
         (128., 128., 128., 30., 30.))
SCALES = (120., 120., 120., 30., 30.)
VARIANTS = ("raw_linear", "adjusted_linear", "raw_expanded", "adjusted_expanded")


@njit(cache=True, fastmath=False)
def normalized_prefix(innovations, phi, prefix):
    """Retain scalar source-operation order and the complete trajectory extrema."""
    path = innovations.copy()
    for index in range(1, len(path)):
        path[index] = path[index - 1] * phi + innovations[index]
    minimum = path.min()
    for index in range(len(path)):
        path[index] = path[index] - minimum
    maximum = path.max()
    result = np.empty(prefix, dtype=np.float64)
    for index in range(prefix):
        divided = path[index] / maximum
        multiplied = 2. * divided
        result[index] = multiplied - 1.
    return result


def literal_angle_ticks(values):
    """Match %0.1f, resolving binary-product half integers with the formatter."""
    values = np.asarray(values, dtype=np.float64)
    scaled = values * 10.
    ticks = np.rint(scaled).astype(np.int64)
    halfway = scaled - np.floor(scaled) == 0.5
    for position in np.flatnonzero(halfway):
        ticks.flat[position] = int(round(float("%0.1f" % values.flat[position]) * 10.))
    return ticks


def commands(normalized, control, mean):
    values = np.asarray(normalized) * SCALES[control] + mean
    if control < 3:
        return values.astype(np.int64).astype(np.float64)
    return literal_angle_ticks(values).astype(np.float64) / 10.


def canonical_recorded_commands(values):
    result = np.asarray(values, dtype=float).copy()
    result[:, :3] = np.rint(result[:, :3])
    result[:, 3:] = np.rint(result[:, 3:] * 10.) / 10.
    return result


def draw_prefixes(control, indices, full_length, prefix, budget):
    values = np.empty((len(indices), prefix), dtype=np.float64)
    for row, index in enumerate(indices):
        seed = np.random.SeedSequence([346, budget, control, int(index)])
        innovations = np.random.default_rng(seed).uniform(-1., 1., full_length)
        values[row] = normalized_prefix(innovations, PHI[control], prefix)
    return values


def observed_simulation_commands(repetition, full_length=4096, prefix=2048):
    columns = []
    for control in range(5):
        seed = np.random.SeedSequence([346, 1000, repetition, control])
        innovations = np.random.default_rng(seed).uniform(-1., 1., full_length)
        values = normalized_prefix(innovations, PHI[control], prefix)
        columns.append(commands(values, control, MEANS[0][control]))
    return np.column_stack(columns)


def nuisance_basis(command_matrix, control, condition=0):
    n = len(command_matrix)
    others = [index for index in range(5) if index != control]
    coordinate = ((command_matrix[:, others] - np.asarray(MEANS[condition])[others])
                  / np.asarray(SCALES)[others])
    columns = [np.ones(n)]
    columns.extend(coordinate[:, index] for index in range(4))
    columns.extend(coordinate[:, left] * coordinate[:, right]
                   for left, right in combinations_with_replacement(range(4), 2))
    positions = np.minimum(15, np.arange(n) * 16 // n)
    columns.extend((positions == block).astype(float) for block in range(1, 16))
    return np.column_stack(columns)


def residuals(command_matrix, responses, control, condition=0):
    values = np.asarray(responses, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    centered = values - values.mean(axis=0)
    basis = nuisance_basis(command_matrix, control, condition)
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    cutoff = max(basis.shape) * np.finfo(float).eps * singular[0]
    orthogonal = left[:, singular > cutoff]
    adjusted = centered - orthogonal @ (orthogonal.T @ centered)
    adjusted -= adjusted.mean(axis=0)
    constant = np.all(values == values[0], axis=0)
    adjusted[:, constant] = 0.
    return centered, adjusted, dict(columns=basis.shape[1], rank=orthogonal.shape[1],
                                    singular_values=singular.tolist(), cutoff=float(cutoff))


def features(command_matrix, control):
    values = np.asarray(command_matrix, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    coordinate = (values - 128.) / 120. if control < 3 else values / 30.
    components = [coordinate, coordinate ** 2]
    if control >= 3:
        angle = values * (np.pi / 90.)
        components.extend((np.sin(angle), np.cos(angle)))
    result = np.stack(components, axis=1)
    result -= result.mean(axis=2, keepdims=True)
    return result


def statistics(command_matrix, control, raw, adjusted):
    return feature_statistics(features(command_matrix, control), raw, adjusted)


def feature_statistics(values, raw, adjusted):
    feature_norm = np.sqrt(np.sum(values * values, axis=2))
    results = []
    for response in (raw, adjusted):
        response_norm = np.sqrt(np.sum(response * response, axis=0))
        denominator = feature_norm[:, :, None] * response_norm[None, None, :]
        numerator = np.abs(values @ response)
        correlation = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.)
        results.append((correlation[:, 0, :], correlation.max(axis=1)))
    return np.stack((results[0][0], results[1][0], results[0][1], results[1][1]), axis=1)


def inclusive_pvalues(observed, reference):
    count = np.sum(np.asarray(reference) >= np.asarray(observed)[None, ...], axis=0)
    return (1. + count) / (len(reference) + 1.), count
