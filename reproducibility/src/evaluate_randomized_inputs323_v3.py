"""Final original-row baseline entry points."""
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from baseline_worker321_v2 import read, save
PROJECT = Path(__file__).resolve().parent


def root_bic(task):
    case = task["case"]
    data = np.load(case["group_paths"][0])
    data = (data - data.mean(0)) / data.std(0, ddof=1)
    n, p = data.shape
    gram = data.T @ data / n
    graph, details = [], []
    for output in case["output_indices"]:
        cache = {}

        def score(parents):
            key = tuple(sorted(parents))
            if key not in cache:
                covariance = gram[np.ix_(key, key)]
                if key and np.linalg.matrix_rank(covariance) < len(key):
                    cache[key] = float("inf")
                else:
                    explained = float(gram[output, list(key)] @ np.linalg.solve(covariance, gram[list(key), output])) if key else 0.
                    variance = gram[output, output] - explained
                    cache[key] = float(n * np.log(max(variance, np.finfo(float).tiny)) + len(key) * np.log(n))
            return cache[key]

        parents = set()
        current = score(parents)
        while True:
            candidates = [(score(parents ^ {root}), root) for root in case["root_indices"]]
            value, root = min(candidates)
            if value >= current - 1e-9:
                break
            parents ^= {root}
            current = value
        graph.extend([root, output] for root in sorted(parents))
        details.append({"output": output, "parents": sorted(parents), "score": current,
                        "evaluated_parent_sets": len(cache)})
    result = {"status": "completed", "method": "Gaussian_BIC_root_domain", "case": case["global_id"],
              "native_edges": graph, "native_undirected_edges": [], "qualified_edges": graph,
              "metadata": {"algorithm": "forward_backward_single_edge_local_search",
                           "parent_domain": "all_recorded_randomized_roots", "maximum_indegree": None,
                           "local_fits": details}, "truth_read": False}
    save(Path(task["output"]), result)
    return result


def fit_baseline(task):
    began = time.perf_counter()
    method, case = task["method"], task["case"]
    output = Path(task["output"])
    if output.exists():
        return read(output)
    try:
        if method == "Gaussian_BIC_root_domain":
            result = root_bic(task)
        elif method == "GES":
            from causallearn.search.ScoreBased.GES import ges
            sys.path.insert(0, str(PROJECT / "round321"))
            from baseline_worker321_v2 import endpoint_edges, qualified
            data = np.load(case["group_paths"][0])
            data = (data - data.mean(0)) / data.std(0, ddof=1)
            native = ges(data, score_func="local_score_BIC", maxP=case["p"] - 1)
            matrix = np.asarray(native["G"].graph)
            directed, undirected = endpoint_edges(matrix, "causal_learn")
            result = {"status": "completed", "method": method, "case": case["global_id"],
                      "native_edges": sorted(map(list, directed)),
                      "native_undirected_edges": sorted(map(list, undirected)),
                      "qualified_edges": sorted(map(list, qualified(directed | undirected, case))),
                      "metadata": {"maxP": case["p"] - 1, "score": float(np.asarray(native["score"]).item())},
                      "truth_read": False}
        else:
            sys.path.insert(0, str(PROJECT / "round321"))
            from baseline_worker321_v2 import fit
            result = fit(task)
        result["development_dataset"] = case["dataset"]
        result["seconds_round323"] = time.perf_counter() - began
    except Exception:
        result = {"status": "ERROR", "method": method, "case": case["global_id"],
                  "traceback": traceback.format_exc(), "seconds_round323": time.perf_counter() - began}
    save(output, result)
    return {"method": method, "case": case["global_id"], "status": result["status"],
            "seconds": result["seconds_round323"]}
