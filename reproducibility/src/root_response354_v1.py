"""Canonical root-response moment frames under target randomization."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT / "round347"), str(PROJECT / "round349")]

import numpy as np
from numba import njit

from joint_moment347_v1 import normalized_columns, response_features
from joint_randomization349_v1 import canonical_features

STATISTICS = ("legacy_mean", "legacy_joint", "root_covariance", "augmented_joint", "covariance_energy", "adaptive_minP")
CANDIDATES = ("augmented_joint", "adaptive_minP", "covariance_energy")


def augmented_responses(controls, raw, adjusted):
    controls = np.asarray(controls, dtype=np.float64)
    raw, adjusted = np.asarray(raw, dtype=np.float64), np.asarray(adjusted, dtype=np.float64)
    if raw.ndim == 1:
        raw, adjusted = raw[:, None], adjusted[:, None]
    heads = raw.shape[1]
    legacy = response_features(raw, adjusted)[:, 2 * heads:]
    roots = normalized_columns(controls)
    residual = adjusted - adjusted.mean(axis=0)
    products = (roots[:, :, None] * residual[:, None, :]).reshape(len(raw), controls.shape[1] * heads)
    return np.ascontiguousarray(np.column_stack((legacy, normalized_columns(products))))


@njit(cache=True, fastmath=False)
def accumulated_frame(features, responses, heads, permutations):
    rows = len(permutations) + 1
    result = np.zeros((rows, 5, heads), dtype=np.float64)
    for row in range(rows):
        cross = np.zeros((features.shape[0], responses.shape[1]), dtype=np.float64)
        for position in range(features.shape[1]):
            source = position if row == 0 else permutations[row - 1, position]
            for feature in range(features.shape[0]):
                value = features[feature, source]
                for response in range(responses.shape[1]):
                    cross[feature, response] += value * responses[position, response]
        for head in range(heads):
            mean, joint, root, energy = 0., 0., 0., 0.
            for feature in range(features.shape[0]):
                mean = max(mean, abs(cross[feature, head]))
                joint = max(joint, abs(cross[feature, head]), abs(cross[feature, head + heads]))
                for component in range(responses.shape[1] // heads):
                    value = cross[feature, component * heads + head]
                    energy += value * value
                    if component >= 2:
                        root = max(root, abs(value))
            result[row, 0, head] = mean
            result[row, 1, head] = joint
            result[row, 2, head] = root
            result[row, 3, head] = max(joint, root)
            result[row, 4, head] = energy
    return result


def adaptive_frame(base):
    rows = len(base)
    heads = base.shape[2]
    adaptive = np.zeros((rows, 1, heads), dtype=np.float64)
    for head in range(heads):
        joint = base[:, 1, head]
        root = base[:, 2, head]
        joint_rank = rows - np.searchsorted(np.sort(joint), joint, side="left")
        root_rank = rows - np.searchsorted(np.sort(root), root, side="left")
        adaptive[:, 0, head] = -np.minimum(joint_rank, root_rank)
    return np.ascontiguousarray(np.concatenate((base, adaptive), axis=1))


def complete_frame(features, responses, heads, permutations):
    return adaptive_frame(accumulated_frame(features, responses, heads, permutations))


def original_ranks(frame):
    return np.sum(frame >= frame[0], axis=0, dtype=np.int64)
