"""Conditional GIES score for the common hardware root domain."""
import numpy as np
from baseline_worker321_v2 import endpoint_edges, qualified
from run_expanded_real_baselines323_v1 import arrays as grouped_arrays


def hardware_gies(task):
    import gies
    from gies.scores import GaussIntL0Pen
    case = task["case"]
    roots, outputs = set(case["root_indices"]), set(case["output_indices"])

    class HardwareScore(GaussIntL0Pen):
        def __init__(self, values):
            super().__init__(values, [[] for _ in values])
            self.interv = [sorted(roots) for _ in values]

        def local_score(self, child, parents):
            if child in roots:
                return 0. if not parents else -np.inf
            if child not in outputs or not set(parents) <= roots:
                return -np.inf
            return super().local_score(child, parents)

        def full_score(self, adjacency):
            return sum(self.local_score(child, set(np.flatnonzero(adjacency[:, child])))
                       for child in range(self.p))

    values, scale = grouped_arrays(case, task["groups"])
    score = HardwareScore(values)
    matrix, value = gies.fit(score)
    assert np.isfinite(value)
    extension = gies.utils.pdag_to_dag(matrix)
    rescored = float(score.full_score(extension))
    assert np.isfinite(rescored) and abs(rescored - value) < 1e-5
    native, undirected = endpoint_edges(np.asarray(matrix), "gies")
    return dict(status="completed", method="GIES", native_edges=sorted(map(list, native)),
                native_undirected_edges=sorted(map(list, undirected)),
                qualified_edges=sorted(map(list, qualified(native | undirected, case))), truth_read=False,
                metadata=dict(score=float(value), extension_score=rescored,
                              targets=[sorted(roots) for _ in values], common_scale=scale.tolist(),
                              parent_domain="hardware_roots", root_score="zero_when_all_groups_intervene",
                              score_extension="inherited_conditional_likelihood_given_controlled_roots"))
