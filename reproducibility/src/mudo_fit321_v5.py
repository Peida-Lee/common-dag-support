"""MuDo-CoM masked optimization and graph mapping."""
import numpy as np


def masked_step(parameters_to_fit,optimizer,covariance,active,identity,batch_loss):
    """Keep saved autograd masks immutable while excluding failed independent paths."""
    import torch
    batch=len(active)
    previous_active=active.clone()
    for value in parameters_to_fit:
        active=active & torch.isfinite(value.detach()).reshape(batch,-1).all(1)
    defaults=[identity[None],torch.ones_like(parameters_to_fit[1]),
              torch.ones_like(parameters_to_fit[2]),torch.zeros_like(parameters_to_fit[3])]
    effective=[torch.where(active.reshape(batch,*([1]*(value.ndim-1))),value,default)
               for value,default in zip(parameters_to_fit,defaults)]
    with torch.no_grad():
        _,info=torch.linalg.inv_ex(identity-effective[3]*(1-identity))
        active=active & (info==0)
    effective=[torch.where(active.reshape(batch,*([1]*(value.ndim-1))),value,default)
               for value,default in zip(parameters_to_fit,defaults)]
    optimizer.zero_grad()
    losses=batch_loss(*effective,covariance)
    active=active & torch.isfinite(losses)
    torch.where(active,losses,torch.zeros_like(losses)).sum().backward()
    for value in parameters_to_fit:
        active=active & torch.isfinite(value.grad).reshape(batch,-1).all(1)
    for value in parameters_to_fit:
        value.grad[~active]=0.
    optimizer.step()
    return active,losses,previous_active & ~active


def graph_from_mixing(mixing,threshold=.3):
    """Use the frozen reciprocal-diagonal assignment and magnitude DAG pruning."""
    from scipy.optimize import linear_sum_assignment
    import networkx as nx
    inverse = np.linalg.inv(mixing)
    rows,columns = linear_sum_assignment(1/np.maximum(abs(inverse),1e-12))
    permutation = np.empty(len(mixing),dtype=int)
    permutation[columns] = rows
    inverse = inverse[permutation]
    diagonal = np.diag(inverse)
    assert np.min(abs(diagonal)) >= 1e-10
    weights = np.eye(len(mixing))-inverse/diagonal[:,None]
    np.fill_diagonal(weights,0.)
    candidates = [(float(abs(weights[child,parent])),parent,child)
                  for child,parent in zip(*np.nonzero(abs(weights)>threshold))]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(mixing)))
    for magnitude,parent,child in sorted(candidates,key=lambda value:(-value[0],value[1],value[2])):
        if not nx.has_path(graph,child,parent):
            graph.add_edge(parent,child)
    return {(int(parent), int(child)) for parent, child in graph.edges()}, weights.T, dict(row_permutation=permutation.tolist(),threshold=threshold,
         assignment='minimize reciprocal absolute diagonal of inverse mixing',
         graph_adapter='magnitude ordered acyclic pruning, fixed before scores')
