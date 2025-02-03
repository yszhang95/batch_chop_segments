import numpy as np
import torch

import timeit

from common_dataset import generate_dataset

def np_loop(X, Ns):
    for i in range(len(Ns)):
        xs = np.linspace(X[i,0], X[i,3], Ns[i]+1)
        ys = np.linspace(X[i,1], X[i,4], Ns[i]+1)
        zs = np.linspace(X[i,2], X[i,5], Ns[i]+1)
        np.stack([
            xs[:-1],
            ys[:-1],
            zs[:-1],
            xs[1:],
            ys[1:],
            zs[1:],
            ], axis=1
        ).reshape(-1, 6)

def torch_loop(X, Ns):
    for i in range(len(Ns)):
        xs = torch.linspace(X[i,0], X[i,3], Ns[i]+1)
        ys = torch.linspace(X[i,1], X[i,4], Ns[i]+1)
        zs = torch.linspace(X[i,2], X[i,5], Ns[i]+1)
        torch.stack([
            xs[:-1],
            ys[:-1],
            zs[:-1],
            xs[1:],
            ys[1:],
            zs[1:],
            ], dim=1
        ).view(-1, 6)

X0, N0 = generate_dataset()
X1 = np.array(X0)
N1 = np.array(N0)
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

n0 = 5
n1 = 5
dt0 = t0.timeit(n0)
dt1 = t1.timeit(n1)
print('torch', dt0/n0, 'sec for ', n0, 'runs')
print('np', dt1/n1, 'sec for ', n1, 'runs')
