"""Common permutation arithmetic and exact rank combination."""

from bisect import bisect_right
from math import prod

import numpy as np
from numba import njit

from joint_moment347_v1 import target_features


STATISTICS = ("adjusted_expanded_mean", "adjusted_expanded_joint")
COMBINATIONS = ("joint_minP", "selected_Fisher", "retained_Fisher")


def canonical_features(target, name):
    target = np.asarray(target, dtype=np.float64)
    ordered = np.sort(target)
    features, metadata = target_features(ordered, name)
    indices = np.searchsorted(ordered, target, side="left")
    return np.ascontiguousarray(features[:, indices]), metadata


@njit(cache=True, fastmath=False)
def reference_frame(features, responses, permutations):
    """Use identical sample accumulation for identity and reference assignments."""
    n = features.shape[1]
    heads = responses.shape[1] // 2
    rows = len(permutations) + 1
    result = np.empty((rows, 2, heads), dtype=np.float64)
    for row in range(rows):
        cross = np.zeros((features.shape[0], responses.shape[1]), dtype=np.float64)
        for position in range(n):
            source = position if row == 0 else permutations[row - 1, position]
            for feature in range(features.shape[0]):
                value = features[feature, source]
                for response in range(responses.shape[1]):
                    cross[feature, response] += value * responses[position, response]
        for head in range(heads):
            mean = 0.
            joint = 0.
            for feature in range(features.shape[0]):
                first = abs(cross[feature, head])
                second = abs(cross[feature, head + heads])
                mean = max(mean, first)
                joint = max(joint, first, second)
            result[row, 0, head] = mean
            result[row, 1, head] = joint
    return result


def combine_ranks(scores, selected, q_numerator=1, q_denominator=1800):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("A finite reference-by-acquisition matrix is required")
    rows, acquisitions = scores.shape
    selected = list(selected)
    if not selected or len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= acquisitions:
        raise ValueError("A nonempty acquisition subset with unique valid indices is required")
    ranks = np.column_stack([rows - np.searchsorted(np.sort(scores[:, group]), scores[:, group], side="left")
                            for group in range(acquisitions)]).astype(np.int64)
    minimum = ranks.min(axis=1)
    legacy = acquisitions * q_denominator * minimum <= rows * q_numerator
    legacy_count = int(legacy.sum())
    bound = acquisitions * ((rows * q_numerator) // (acquisitions * q_denominator))
    assert legacy_count <= bound <= (rows * q_numerator) // q_denominator
    products = [prod(int(value) for value in values) for values in ranks[:, selected]]
    ordered = sorted(products)
    first = sorted(value for value, active in zip(products, legacy) if active)
    second = sorted(value for value, active in zip(products, legacy) if not active)
    min_counts = np.searchsorted(np.sort(minimum), minimum, side="right")
    fisher_counts = np.asarray([bisect_right(ordered, value) for value in products], dtype=np.int64)
    retained_counts = np.asarray([bisect_right(first, value) if active
                                  else legacy_count + bisect_right(second, value)
                                  for value, active in zip(products, legacy)], dtype=np.int64)
    counts = np.column_stack((min_counts, fisher_counts, retained_counts))
    assert np.all((1 <= counts) & (counts <= rows))
    assert np.all(min_counts <= acquisitions * minimum)
    assert np.all(q_denominator * retained_counts[legacy] <= rows * q_numerator)
    metadata = dict(reference_rows=rows, acquisitions=acquisitions, selected_acquisitions=selected,
                    observed_acquisition_ranks=ranks[0].tolist(), legacy_reference_rows=legacy_count,
                    legacy_reference_bound=int(bound), available_rejection_rows=int((rows * q_numerator) // q_denominator),
                    observed_legacy_rejection=bool(legacy[0]), q_numerator=q_numerator, q_denominator=q_denominator)
    return counts, metadata
