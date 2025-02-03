import numba as nb
import numpy as np
import torch

import timeit

from common_dataset import generate_dataset

def chop_np(X0X1, Ns):
    batch_size = X0X1.shape[0]

    # Find the max segment count to unify tensor sizes
    max_size = Ns.max()
    N_LIMIT = 1000
    MIN_STEP = 1/N_LIMIT
    assert max_size < N_LIMIT

    # Create a common index tensor (0 to max_size)
    idxs = np.arange(0, max_size+0.1, 1, dtype=np.float32)  # Shape: (max_size + 1,)

    # Normalize indices based on Ns (broadcasting)
    frac_steps = idxs[:, None] / Ns[None, :]  # Shape: (max_size+1, batch_size)
    frac_steps = frac_steps.T  # Shape: (batch_size, max_size+1)

    # Mask to ignore extra indices for each row
    mask = frac_steps < (1+MIN_STEP)

    # Compute interpolated values using broadcasting
    xs = (1 - frac_steps) * X0X1[:, 0, None] + frac_steps * X0X1[:, 3, None]
    ys = (1 - frac_steps) * X0X1[:, 1, None] + frac_steps * X0X1[:, 4, None]
    zs = (1 - frac_steps) * X0X1[:, 2, None] + frac_steps * X0X1[:, 5, None]

    # Apply mask
    xs, ys, zs = xs * mask, ys * mask, zs * mask

    mask = frac_steps < (1-MIN_STEP) # discard the last point as the lenght of output is less than the lenght of linspace by 1.

    # Stack and reshape for output format
    output = np.stack([xs[:, :-1], ys[:, :-1], zs[:, :-1], xs[:, 1:], ys[:, 1:], zs[:, 1:]], axis=-1, dtype=np.float32)
    return output[mask[:, :-1]].reshape(-1, 6)  # Remove invalid entries

@nb.njit
def chop_nb(X0X1, Ns):
    # Example large N
    # Compute the total size of the output array
    total_size = np.sum(Ns)

    # Preallocate a single output array
    result = np.empty((total_size, 6), dtype=np.float32)

    # Fill it using a loop but with direct indexing (avoiding repeated concatenation)
    idx = 0
    for i in range(len(X0X1)):
        size = Ns[i]
        for j in range(3):
            temp = np.linspace(X0X1[i,j], X0X1[i,j+3], size+1)
            result[idx:idx+size,j] = temp[:-1]
            result[idx:idx+size,j+3] = temp[1:]
        idx += size  # Move the index forward
    return result

def chop_torch_optimized(X0X1, Ns):
    batch_size = X0X1.shape[0]
    device = X0X1.device

    # Find the max segment count to unify tensor sizes
    max_size = Ns.max().item()
    N_LIMIT = 1000
    MIN_STEP = 1/N_LIMIT
    assert max_size + 1 < N_LIMIT

    # Create a common index tensor (0 to max_size)
    idxs = torch.arange(max_size + 0.1, device=device).float()  # Shape: (max_size + 1,)

    # Normalize indices based on Ns (broadcasting)
    frac_steps = idxs[:, None] / Ns[None, :]  # Shape: (max_size+1, batch_size)
    frac_steps = frac_steps.T  # Shape: (batch_size, max_size+1)

    # Mask to ignore extra indices for each row
    mask = frac_steps < (1+MIN_STEP)

    # Compute interpolated values using broadcasting
    xs = (1 - frac_steps) * X0X1[:, 0, None] + frac_steps * X0X1[:, 3, None]
    ys = (1 - frac_steps) * X0X1[:, 1, None] + frac_steps * X0X1[:, 4, None]
    zs = (1 - frac_steps) * X0X1[:, 2, None] + frac_steps * X0X1[:, 5, None]

    # Apply mask
    xs, ys, zs = xs * mask, ys * mask, zs * mask

    mask = frac_steps < (1-MIN_STEP) # discard the last point as the lenght of output is less than the lenght of linspace by 1.

    # Stack and reshape for output format
    output = torch.stack([xs[:, :-1], ys[:, :-1], zs[:, :-1], xs[:, 1:], ys[:, 1:], zs[:, 1:]], dim=-1)
    return output[mask[:, :-1]].view(-1, 6)  # Remove invalid entries

def main():
    torch.manual_seed(1024)
    device = 'cpu'
    X0 = torch.rand((8_000_000, 3), requires_grad=False, device=device)
    X1 = (torch.rand((8_000_000, 3), requires_grad=False, device=device) + 0.5) * 2
    step_limit = 1
    X0X1 = torch.cat([X0,X1], dim=1)
    LdX = torch.linalg.norm(X1-X0, dim=1)
    Ns = (LdX//step_limit + 1).to(torch.int32)

    o1 = chop_torch_optimized(X0X1, Ns)

    t1 = timeit.Timer(stmt='chop_torch_optimized(X0X1, Ns)', setup='from __main__ import chop_torch_optimized', globals={'X0X1' : X0X1, 'Ns' : Ns})

    X0X1 = np.array(X0X1)
    Ns = np.array(Ns)
    t2 = timeit.Timer(stmt='chop_nb(X0X1, Ns)', setup='from __main__ import chop_nb', globals={'X0X1' : X0X1, 'Ns' : Ns})
    t3 = timeit.Timer(stmt='chop_np(X0X1, Ns)', setup='from __main__ import chop_np', globals={'X0X1' : X0X1, 'Ns' : Ns})

    o2 = chop_nb(X0X1, Ns)
    o3 = chop_np(X0X1, Ns)

    dt1 = t1.timeit(10)
    dt2 = t2.timeit(10)
    dt3 = t3.timeit(10)
    print('torch', dt1)
    print('nb', dt2)
    print('np', dt3)

    o1 = np.array(o1)
    print(o1.dtype)
    print(o2.dtype)
    print(o3.dtype)

    assert np.allclose(o2, o3)
    assert np.allclose(o1, o3)


main()
