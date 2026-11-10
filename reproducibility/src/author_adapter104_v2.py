"""Batched published MuDo-CoM objective and initialization."""
import numpy as np
import torch
from torch import nn


def batch_loss(A,D,sigma,Lfree,cov):
    p=A.shape[-1];I=torch.eye(p,device=A.device,dtype=A.dtype);L=Lfree*(1-I)
    W=torch.inverse(I-L);S=torch.diag_embed(sigma);M=W@(S@S)@W.transpose(-1,-2)
    latent=D[:,:,:,None]*M[:,None,:,:]*D[:,:,None,:]
    predicted=A[:,None,:,:]@latent@A[:,None,:,:].transpose(-1,-2)
    return ((cov-predicted)**2).sum(dim=(-3,-2,-1))


def parameters(batch,p,T,device,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    init=[]
    for _ in range(batch):
        init.append((torch.rand(p,p,device=device),torch.rand(T,p,device=device)*.8+.2,
            torch.rand(p,device=device)*.8+.2,torch.zeros(p,p,device=device)))
    return [nn.Parameter(torch.stack([x[k] for x in init])) for k in range(4)]
