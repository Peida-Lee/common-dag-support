"""Published MuDo-CoM fit on shared standardized row groups."""
from pathlib import Path
import sys
import traceback
import numpy as np
from baseline_worker321_v2 import read, save, qualified
PROJECT = Path(__file__).resolve().parent


def arrays(case, groups):
    pooled = np.load(case["group_paths"][0])
    center, scale = pooled.mean(0), pooled.std(0, ddof=1)
    result = []
    for group in groups:
        with np.load(group["stored"]) as data:
            positions = [group["names"].index(name) for name in case["variables"]]
            result.append(np.asarray((data["values"][:, positions] - center) / scale))
    return result, scale


def fit_mudo(task):
    import torch
    from mudo_fit321_v5 import masked_step, graph_from_mixing
    sys.path.insert(0, str(PROJECT / "round104"))
    from author_adapter104_v2 import parameters, batch_loss
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.2)
    case = task["case"]
    values, scale = arrays(case, task["groups"])
    moments = np.stack([np.cov(value, rowvar=False) for value in values])
    covariance = torch.tensor(moments[None], dtype=torch.float32, device="cuda").expand(3, -1, -1, -1)
    fitted = parameters(3, case["p"], len(values), "cuda", 3238000 + case["p"])
    optimizer = torch.optim.Adam(fitted, lr=.005)
    active = torch.ones(3, dtype=torch.bool, device="cuda")
    identity = torch.eye(case["p"], device="cuda")
    for step in range(50000):
        active, losses, failed = masked_step(fitted, optimizer, covariance, active, identity, batch_loss)
    destination = Path(task["output"]).parent
    np.savez_compressed(destination / "fitted_parameters.npz",
                        **{name: value.detach().cpu().numpy() for name, value in
                           zip(("A", "D", "sigma", "Lnoise"), fitted)}, active=active.cpu().numpy())
    candidates, failures = [], []
    for restart in range(3):
        try:
            if not active[restart] or not all(bool(torch.isfinite(value[restart]).all()) for value in fitted):
                raise ArithmeticError("Nonfinite independent optimization path")
            with torch.no_grad():
                loss = float(batch_loss(*[value[restart:restart + 1] for value in fitted], covariance[:1]).cpu()[0])
            if not np.isfinite(loss):
                raise ArithmeticError("Nonfinite final objective")
            graph, weights, detail = graph_from_mixing(fitted[0][restart].detach().cpu().numpy())
            candidates.append((loss, restart, graph, weights, detail))
        except Exception:
            failures.append({"restart": restart, "traceback": traceback.format_exc()})
    save(destination / "independent_path_failures.json", failures)
    if not candidates:
        raise ArithmeticError("All MuDo-CoM paths require repair. See saved parameter paths.")
    loss, restart, graph, weights, detail = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return {"status": "completed", "method": "MuDo_CoM", "case": case["global_id"],
            "native_edges": sorted(map(list, graph)), "native_undirected_edges": [],
            "qualified_edges": sorted(map(list, qualified(graph, case))), "weights": np.asarray(weights).tolist(),
            "metadata": {"iterations": 50000, "restarts": 3, "selected_restart": restart,
                         "loss": loss, "failed_paths": failures, "graph_adapter": detail,
                         "input_environment_count": len(values), "common_scale": scale.tolist()},
            "truth_read": False}
