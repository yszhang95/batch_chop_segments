import numpy as np
import torch

import timeit
import torch.utils.benchmark as benchmark

from common_dataset import generate_dataset

def np_loop(X, Ns):
    for i in range(len(Ns)):
        xs = np.linspace(X[i,0], X[i,3], Ns[i]+1)
        ys = np.linspace(X[i,1], X[i,4], Ns[i]+1)
        zs = np.linspace(X[i,2], X[i,5], Ns[i]+1)
        xs[:-1]
        ys[:-1]
        zs[:-1]
        xs[1:]
        ys[1:]
        zs[1:]

def torch_loop(X, Ns):
    for i in range(len(Ns)):
        xs = torch.linspace(X[i,0], X[i,3], Ns[i]+1, device=Ns.device)
        ys = torch.linspace(X[i,1], X[i,4], Ns[i]+1, device=Ns.device)
        zs = torch.linspace(X[i,2], X[i,5], Ns[i]+1, device=Ns.device)
        xs[:-1]
        ys[:-1]
        zs[:-1]
        xs[1:]
        ys[1:]
        zs[1:]

X0, N0 = generate_dataset()
X1 = np.array(X0)
N1 = np.array(N0)
X2 = X0.to('cuda:0')
N2 = N0.to('cuda:0')

t0 = timeit.Timer(
    stmt = 'torch_loop(X, Ns)',
    setup = 'from __main__ import torch_loop',
    globals = {
        'X' : X0,
        'Ns' : N0,
    }
)

t1 = timeit.Timer(
    stmt = 'np_loop(X, Ns)',
    setup = 'from __main__ import np_loop',
    globals = {
        'X' : X1,
        'Ns' : N1,
    }
)

t2 = benchmark.Timer(
    stmt = 'torch_loop(X, Ns)',
    setup = 'from __main__ import torch_loop',
    globals = {
        'X' : X2,
        'Ns' : N2,
    }
)

n0 = 5
n1 = 5
n2 = 5
dt0 = t0.timeit(n0)
dt1 = t1.timeit(n1)
dt2 = t2.blocked_autorange()
print('torch', dt0/n0, 'sec for ', n0, 'runs')
print('np', dt1/n1, 'sec for ', n1, 'runs')
print('cuda', dt2)
