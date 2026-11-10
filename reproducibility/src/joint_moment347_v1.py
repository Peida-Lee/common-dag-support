"""Permutation-invariant target features and joint residual-moment evidence."""

import numpy as np


VARIANTS = ("raw_linear_mean", "adjusted_linear_mean", "raw_expanded_mean",
            "adjusted_expanded_mean", "raw_expanded_joint", "adjusted_expanded_joint")


def target_features(target, name):
    target = np.asarray(target, dtype=np.float64)
    ordered = np.sort(target)
    lower, upper = ordered[0], ordered[-1]
    scale = upper - lower
    coordinate = 2. * (target - lower) / scale - 1. if scale else np.zeros_like(target)
    ordered_coordinate = 2. * (ordered - lower) / scale - 1. if scale else np.zeros_like(ordered)
    values = [coordinate, coordinate ** 2]
    canonical = [ordered_coordinate, ordered_coordinate ** 2]
    if name.startswith("pol_"):
        values.extend((np.sin(target * (np.pi / 90.)), np.cos(target * (np.pi / 90.))))
        canonical.extend((np.sin(ordered * (np.pi / 90.)), np.cos(ordered * (np.pi / 90.))))
    values = np.column_stack(values)
    canonical = np.column_stack(canonical)
    center = canonical.mean(axis=0)
    centered_canonical = canonical - center
    norm = np.sqrt(np.sum(centered_canonical * centered_canonical, axis=0))
    constant = np.all(canonical == canonical[0], axis=0)
    norm[constant] = 0.
    normalized = np.divide(values - center, norm, out=np.zeros_like(values), where=norm > 0.)
    return np.ascontiguousarray(normalized.T), dict(lower=float(lower), upper=float(upper),
         canonical_center=center.tolist(), canonical_norm=norm.tolist(), constant=constant.tolist())


def normalized_columns(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0)
    constant = np.all(values == values[0], axis=0)
    norm = np.sqrt(np.sum(centered * centered, axis=0))
    norm[constant] = 0.
    return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0.)


def response_features(raw, adjusted):
    raw = np.asarray(raw, dtype=float)
    adjusted = np.asarray(adjusted, dtype=float)
    if raw.ndim == 1:
        raw, adjusted = raw[:, None], adjusted[:, None]
    raw = raw - raw.mean(axis=0)
    adjusted = adjusted - adjusted.mean(axis=0)
    return np.ascontiguousarray(np.column_stack((normalized_columns(raw), normalized_columns(raw ** 2),
                                                normalized_columns(adjusted), normalized_columns(adjusted ** 2))))


def statistics(feature_batch, responses):
    feature_batch = np.asarray(feature_batch, dtype=float)
    if feature_batch.ndim == 2:
        feature_batch = np.ascontiguousarray(feature_batch[None, :, :])
    cross = np.abs(feature_batch @ responses)
    assert np.isfinite(cross).all()
    outputs = responses.shape[1] // 4
    raw = cross[:, :, :outputs]
    adjusted = cross[:, :, 2 * outputs:3 * outputs]
    shape = (len(cross), cross.shape[1], 2, outputs)
    raw_joint = cross[:, :, :2 * outputs].reshape(shape).max(axis=(1, 2))
    adjusted_joint = cross[:, :, 2 * outputs:].reshape(shape).max(axis=(1, 2))
    return np.stack((raw[:, 0, :], adjusted[:, 0, :], raw.max(axis=1), adjusted.max(axis=1),
                     raw_joint, adjusted_joint), axis=1)


def permuted_statistics(features, responses, permutations):
    permuted = np.ascontiguousarray(features[:, permutations].transpose(1, 0, 2))
    return statistics(permuted, responses)


def pvalues(observed, references):
    count = np.sum(references >= observed[None, ...], axis=0)
    return (count + 1.) / (len(references) + 1.), count
