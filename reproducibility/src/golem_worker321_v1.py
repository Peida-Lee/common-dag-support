"""Native GOLEM on an equivalent square root of the original data Gram matrix."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CASTLE_BACKEND'] = 'pytorch'

from pathlib import Path
import contextlib
import time
import traceback
import numpy as np
from baseline_worker321_v2 import edges, qualified, read, save

BASE = Path(__file__).resolve().parent


def compact_rows(data):
    gram = data.T @ data
    return np.linalg.cholesky(gram).T


def check_equivalence():
    import torch
    from castle.algorithms.gradient.notears.torch.golem_utils.golem_model import GolemModel
    torch.set_num_threads(1)
    plan = read(BASE/'baseline_plan321_v1.json')
    case = plan['cases'][0]
    original = np.load(case['group_paths'][0], mmap_mode='r')[:2048]
    data = (original - original.mean(0)) / original.std(0, ddof=1)
    root = compact_rows(data)
    records = []
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)
        for equal_variances in (True, False):
            initial = np.random.default_rng(321701).normal(0, .02, (case['p'], case['p']))
            initial = initial.astype(np.float64 if dtype == torch.float64 else np.float32)
            models = [GolemModel(len(x), case['p'], .02, 5., equal_variances,
                                B_init=initial.copy(), device=torch.device('cpu'))
                      for x in (data, root)]
            arrays = [torch.tensor(x, dtype=dtype) for x in (data, root)]
            optimizers = [torch.optim.Adam(model.parameters(), lr=.001) for model in models]
            for model, array in zip(models, arrays):
                model(array)
                model.score.backward()
            values = [float(model.score.detach()) for model in models]
            value_error = abs(values[0]-values[1]) / max(1., abs(values[0]))
            gradient_error = float((models[0]._B.grad-models[1]._B.grad).abs().max())
            gradient_scale = max(1., float(models[0]._B.grad.abs().max()))
            for step in range(10):
                for model, array, optimizer in zip(models, arrays, optimizers):
                    optimizer.zero_grad()
                    model(array)
                    model.score.backward()
                    optimizer.step()
            update_error = float((models[0]._B-models[1]._B).abs().max().detach())
            tolerance = 1e-10 if dtype == torch.float64 else 1e-4
            assert value_error < tolerance
            assert gradient_error/gradient_scale < tolerance
            assert update_error < tolerance
            records.append(dict(dtype=str(dtype), equal_variances=equal_variances,
                                relative_objective_error=value_error,
                                scaled_gradient_error=gradient_error/gradient_scale,
                                Adam10_parameter_error=update_error))
    torch.set_default_dtype(torch.float32)
    receipt = dict(status='passed', checks=records, truth_read=False,
                   source_rows=len(data), compact_rows=len(root), source_n_unchanged=True)
    save(BASE/'GOLEM_equivalence321_v1.json', receipt)
    return receipt


def fit(task):
    case = task['case']
    pilot = task.get('pilot', False)
    output = Path(task['output'])
    if output.exists():
        return read(output)
    began = time.perf_counter()
    try:
        import torch
        from castle.algorithms import GOLEM
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        assert read(BASE/'GOLEM_equivalence321_v1.json')['status'] == 'passed'
        n = min(case['n'], 1024) if pilot else case['n']
        original = np.load(case['group_paths'][0], mmap_mode='r')[:n]
        center, scale = original.mean(0), original.std(0, ddof=1)
        data = np.asarray((original-center)/scale, dtype=np.float64)
        root = compact_rows(data)
        output.parent.mkdir(parents=True, exist_ok=True)
        variants = {}
        initial = None
        with output.with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for flavor, penalty in [('EV', .02), ('NV', .002)]:
                model = GOLEM(B_init=initial, lambda_1=penalty, lambda_2=5.,
                              equal_variances=flavor=='EV', learning_rate=.001,
                              num_iter=200 if pilot else 100000,
                              checkpoint_iter=100 if pilot else 5000,
                              seed=3217000+case['global_id'], graph_thres=.3, device_type='cpu')
                model.learn(root)
                initial = np.asarray(model.weight_causal_matrix, dtype=np.float32).copy()
                assert np.all(np.isfinite(initial))
                raw = edges(np.asarray(model.causal_matrix))
                variants[flavor] = dict(weights=initial.tolist(), native_edges=sorted(map(list, raw)),
                                        qualified_edges=sorted(map(list, qualified(raw, case))))
        result = dict(status='completed', method='GOLEM', case=case['global_id'],
                      study=case['study'], template=case['template'], n=n, p=case['p'], pilot=pilot,
                      native_edges=variants['NV']['native_edges'], native_undirected_edges=[],
                      qualified_edges=variants['NV']['qualified_edges'], weights=initial.tolist(),
                      metadata=dict(variants=variants, primary='NV', NV_warm_start='EV',
                                    original_package='gcastle==1.0.4',
                                    iterations_per_stage=200 if pilot else 100000,
                                    compact_Gram_factor=True, source_sample_size=n,
                                    compressed_rows=len(root)),
                      seconds=time.perf_counter()-began, truth_read=False, input_groups=[0],
                      standardization=dict(center=center.tolist(), scale=scale.tolist(), source='control'))
    except Exception:
        result = dict(status='ERROR', method='GOLEM', case=case['global_id'], pilot=pilot,
                      traceback=traceback.format_exc(), seconds=time.perf_counter()-began, truth_read=False)
    save(output, result)
    return result
