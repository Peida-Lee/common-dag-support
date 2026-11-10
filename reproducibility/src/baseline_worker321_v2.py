"""Native baseline calls on original rows; graph truth is never an input."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CASTLE_BACKEND']='pytorch'
os.environ['WANDB_MODE']='disabled'
from pathlib import Path
import json,itertools,time,traceback,contextlib,sys
import numpy as np

BASE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def edges(matrix):
    return {(int(i),int(j)) for i,j in zip(*np.nonzero(matrix)) if i!=j}


def endpoint_edges(matrix, convention):
    """Preserve directed and unoriented edges under the native package convention."""
    directed = set()
    undirected = set()
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            left, right = matrix[i, j], matrix[j, i]
            if convention == 'causal_learn':
                if (left, right) == (-1, 1):
                    directed.add((i, j))
                elif (left, right) == (1, -1):
                    directed.add((j, i))
                elif left or right:
                    undirected.add((i, j))
            elif left and right:
                undirected.add((i, j))
            elif left:
                directed.add((i, j))
            elif right:
                directed.add((j, i))
    return directed, undirected


def qualified(raw,case):
    mandatory=set(map(tuple,case['mandatory']))
    allowed=mandatory|{(q['u'],q['w']) for q in case['queries']}
    skeleton={frozenset(e) for e in raw}
    return mandatory|{e for e in allowed if frozenset(e) in skeleton}


def exact_bic(X,case):
    """The global BIC optimum on U: U is acyclic and each free edge has its own child."""
    n,p=X.shape;S=X.T@X/n;B=np.zeros((p,p));mandatory=set(map(tuple,case['mandatory']))
    optional={(q['u'],q['w']) for q in case['queries']}
    details=[]
    for j in range(p):
        fixed=sorted(u for u,w in mandatory if w==j)
        free=sorted(u for u,w in optional if w==j)
        candidates=[]
        for size in range(len(free)+1):
            for selected in itertools.combinations(free,size):
                pa=sorted(set(fixed)|set(selected))
                if len(pa)>2:continue
                beta=np.linalg.solve(S[np.ix_(pa,pa)],S[pa,j]) if pa else np.empty(0)
                variance=S[j,j]-float(S[j,pa]@beta) if pa else S[j,j]
                assert variance>0
                score=n*np.log(variance)+len(pa)*np.log(n)
                candidates.append((float(score),pa,beta))
        score,pa,beta=min(candidates,key=lambda r:(r[0],r[1]))
        B[pa,j]=beta
        details.append(dict(node=j,parents=pa,score=score,candidates=len(candidates)))
    return B,dict(exact_qualified_BIC=True,local_fits=details)


def fit(task):
    case=task['case'];method=task['method'];pilot=task.get('pilot',False)
    output=Path(task['output']);began=time.perf_counter()
    if output.exists():return read(output)
    try:
        n=min(case['n'],1024) if pilot else case['n']
        original=np.load(case['group_paths'][0],mmap_mode='r')[:n]
        center=np.mean(original,axis=0);scale=np.std(original,axis=0,ddof=1)
        assert np.all(scale>0)
        X=np.asarray((original-center)/scale,dtype=np.float64)
        p=X.shape[1];seed=3214000+100*case['global_id']+task.get('seed_index',0)
        np.random.seed(seed)
        metadata={};weights=None;raw=set();undirected=set()
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if method=='DAGMA':
                from dagma.linear import DagmaLinear
                model=DagmaLinear(loss_type='l2',verbose=False)
                kwargs=dict(lambda1=.03,w_threshold=.3)
                if pilot:kwargs.update(T=2,s=[1.,.9],warm_iter=100,max_iter=100,checkpoint=25)
                weights=np.asarray(model.fit(X,**kwargs));raw=edges(weights)
                metadata=dict(parameters=kwargs,original_package='dagma==1.1.1')
            elif method=='GOLEM':
                import torch
                torch.set_num_threads(1);torch.manual_seed(seed)
                from castle.algorithms import GOLEM
                results={};initial=None
                for flavor,penalty in [('EV',.02),('NV',.002)]:
                    model=GOLEM(B_init=initial,lambda_1=penalty,lambda_2=5.,equal_variances=flavor=='EV',
                         learning_rate=.001,num_iter=200 if pilot else 100000,checkpoint_iter=100 if pilot else 5000,
                         seed=seed,graph_thres=.3,device_type='cpu')
                    model.learn(X)
                    initial=np.asarray(model.weight_causal_matrix,dtype=float)
                    assert np.all(np.isfinite(initial))
                    results[flavor]=dict(weights=initial.tolist(),native_edges=sorted(map(list,edges(np.asarray(model.causal_matrix)))),
                         lambda_1=penalty)
                weights=initial;raw=set(map(tuple,results['NV']['native_edges']))
                metadata=dict(variants=results,primary='NV',NV_warm_start='EV',original_package='gcastle==1.0.4',iterations=200 if pilot else 100000)
            elif method=='PC':
                from causallearn.search.ConstraintBased.PC import pc
                from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
                from causallearn.graph.GraphNode import GraphNode
                allowed=set(map(tuple,case['mandatory']))|{(q['u'],q['w']) for q in case['queries']}
                nodes=[GraphNode('v'+str(j)) for j in range(p)];bk=BackgroundKnowledge()
                for i,j in itertools.permutations(range(p),2):
                    if (i,j) not in allowed:bk.add_forbidden_by_node(nodes[i],nodes[j])
                cg=pc(X,alpha=.05,stable=True,indep_test='fisherz',max_k=2,
                      background_knowledge=bk,node_names=[x.get_name() for x in nodes],show_progress=False)
                matrix=np.asarray(cg.G.graph)
                raw,undirected=endpoint_edges(matrix,'causal_learn')
                metadata=dict(endpoint_matrix=matrix.tolist(),alpha=.05,max_k=2,qualified_background_knowledge=True,original_package='causal-learn==0.1.4.8')
            elif method=='GES':
                from causallearn.search.ScoreBased.GES import ges
                result=ges(cov=np.cov(X,rowvar=False,bias=True),n=n,score_func='local_score_BIC',maxP=2)
                matrix=np.asarray(result['G'].graph)
                raw,undirected=endpoint_edges(matrix,'causal_learn')
                metadata=dict(endpoint_matrix=matrix.tolist(),score=float(np.asarray(result['score']).reshape(-1)[0]),maxP=2,original_package='causal-learn==0.1.4.8')
            elif method=='GIES':
                import gies
                from gies.scores import GaussIntL0Pen
                class CappedScore(GaussIntL0Pen):
                    def local_score(self,j,pa):
                        return -np.inf if len(pa)>2 else super().local_score(j,pa)
                arrays=[]
                for path in case['group_paths']:
                    x=np.load(path,mmap_mode='r')[:n]
                    arrays.append(np.asarray((x-center)/scale,dtype=np.float64))
                targets=[[]]+[[j] for j in range(p)]
                score=CappedScore(arrays,targets)
                matrix,value=gies.fit(score)
                weights=np.asarray(matrix,dtype=float)
                raw,undirected=endpoint_edges(weights,'gies')
                metadata=dict(score=float(value),max_indegree=2,groups=p+1,intervention_model='perfect_intervention_working_reference',original_package='gies==0.0.3')
            elif method=='HC_BIC':
                weights,metadata=exact_bic(X,case);raw=edges(weights)
                metadata.update(original_method='Gaussian BIC with qualified parent domains',global_optimum_under_U=True)
            elif method=='DirectLiNGAM':
                from lingam import DirectLiNGAM
                ancestry=np.asarray(case['ancestors'],dtype=int)
                prior=ancestry.copy();np.fill_diagonal(prior,-1)
                model=DirectLiNGAM(random_state=seed,prior_knowledge=prior)
                model.fit(X)
                weights=np.asarray(model.adjacency_matrix_,dtype=float).T
                raw=edges(weights);metadata=dict(original_package='lingam==1.13.0',prior='certified_ancestry',native_orientation='child_by_parent; transposed')
            elif method=='NOTEARS':
                from castle.algorithms import Notears
                model=Notears(lambda1=.1,loss_type='l2',max_iter=10 if pilot else 100,h_tol=1e-8,rho_max=1e16,w_threshold=.3)
                model.learn(X);weights=np.asarray(model.weight_causal_matrix,dtype=float)
                raw=edges(np.asarray(model.causal_matrix));metadata=dict(original_package='gcastle==1.0.4',lambda1=.1,w_threshold=.3,max_iter=10 if pilot else 100)
            else:raise ValueError('unknown method '+method)
        if weights is not None:assert weights.shape==(p,p) and np.all(np.isfinite(weights))
        qedges=qualified(raw|undirected,case)
        import networkx as nx
        native_graph=nx.DiGraph()
        native_graph.add_nodes_from(range(p));native_graph.add_edges_from(raw)
        result=dict(status='completed',method=method,case=case['global_id'],study=case['study'],template=case['template'],n=n,p=p,pilot=pilot,
                    native_edges=sorted(map(list,raw)),native_undirected_edges=sorted(map(list,undirected)),
                    native_directed_part_acyclic=nx.is_directed_acyclic_graph(native_graph),
                    native_graph_type='CPDAG' if method in ['PC','GES','GIES'] else 'directed',
                    qualified_edges=sorted(map(list,qedges)),weights=None if weights is None else weights.tolist(),
                    metadata=metadata,seconds=time.perf_counter()-began,input_role='original_rows',truth_read=False,
                    standardization=dict(center=center.tolist(),scale=scale.tolist(),source='control'),
                    input_groups=list(range(p+1)) if method=='GIES' else [0])
    except Exception:
        result=dict(status='ERROR',method=method,case=case['global_id'],pilot=pilot,traceback=traceback.format_exc(),seconds=time.perf_counter()-began,truth_read=False)
    save(output,result)
    return result
