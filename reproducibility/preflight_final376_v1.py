"""Check execution equivalence before sealing the final replay."""

from pathlib import Path
import os
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "src"))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CASTLE_BACKEND"] = "pytorch"
os.environ["WANDB_MODE"] = "disabled"
os.environ["NUMBA_CACHE_DIR"] = str(PACKAGE / "environment/numba_cache")

import json
from itertools import permutations
import numpy as np
import runtime346_v1 as runtime


def light_equivalence():
    from root_response354_v1 import accumulated_frame, augmented_responses
    from joint_randomization349_v1 import canonical_features, reference_frame
    from joint_moment347_v1 import response_features
    rng = np.random.default_rng(3760001)
    checks = []
    for name in ("red", "pol_1"):
        target = np.array([0., 0., 1., 2., 3.])
        raw = rng.normal(size=(5, 3))
        raw[:, 2] = 1.
        adjusted = rng.normal(size=(5, 3))
        adjusted[:, 2] = 0.
        controls = rng.normal(size=(5, 4))
        feature, _ = canonical_features(target, name)
        orders = np.asarray(list(permutations(range(5))), dtype=np.uint16)
        augmented = augmented_responses(controls, raw, adjusted)
        legacy = np.ascontiguousarray(response_features(raw, adjusted)[:, 6:])
        full = accumulated_frame(feature, augmented, 3, orders)
        reduced = reference_frame(feature, legacy, orders)
        assert np.array_equal(full[:, 1], reduced[:, 1])
        checks.append(dict(target=name, rows=len(full), bitwise_equal=True))
    return checks


def gram_equivalence():
    import golem_worker321_v1 as golem
    folder = PACKAGE / "specifications/verification"
    folder.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(3760002).normal(size=(128, 4))
    path = folder / "gram_fixture.npy"
    np.save(path, values)
    runtime.save(folder / "baseline_plan321_v1.json", dict(cases=[dict(p=4, group_paths=[str(path)])]))
    golem.BASE = folder
    result = golem.check_equivalence()
    # The gate is part of the final implementation check. Its fixture is regenerated here.
    path.unlink()
    (folder / "baseline_plan321_v1.json").unlink()
    return result


def main():
    report = dict(status="passed", light_legacy_equivalence=light_equivalence(), gram_equivalence=gram_equivalence())
    import torch
    assert torch.__version__ == "2.5.1+cu121"
    assert torch.cuda.is_available()
    import pandas as pd
    import scipy
    import numba
    assert np.__version__ == "1.26.4"
    assert pd.__version__ == "1.5.3"
    assert scipy.__version__ == "1.13.1"
    assert numba.__version__ == "0.59.1"
    report.update(versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                                numba=numba.__version__, torch=torch.__version__),
                  cuda_device=torch.cuda.get_device_name(0), beijing=runtime.stamp(), measured_model_fits=0)
    runtime.save(PACKAGE / "specifications/verification/preflight.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
