import numpy as np
import numba as nb
import timeit

from common_dataset import generate_dataset

# numpy based

def chop_np(X0X1, Ns):
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

def chop_np2(X0X1, Ns):
    output = []
    for i in range(len(X0X1)):
        size = Ns[i]
        xs = np.linspace(X0X1[i,0], X0X1[i,3], size+1);
        ys = np.linspace(X0X1[i,1], X0X1[i,4], size+1);
        zs = np.linspace(X0X1[i,2], X0X1[i,5], size+1);
        output.append(np.stack([xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]], axis=1).reshape(-1, 6))
    return np.concatenate(output, axis=0)

def chop_np_optimized(X0X1, Ns):
    batch_size = X0X1.shape[0]

    # Find the max segment count to unify tensor sizes
    max_size = Ns.max()
    N_LIMIT = 1000
    MIN_STEP = 1/N_LIMIT
    assert max_size < N_LIMIT

    # Create a common index tensor (0 to max_size)
    idxs = np.arange(0, max_size+0.1, 1)  # Shape: (max_size + 1,)

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
    output = np.stack([xs[:, :-1], ys[:, :-1], zs[:, :-1], xs[:, 1:], ys[:, 1:], zs[:, 1:]], axis=-1)
    return output[mask[:, :-1]].reshape(-1, 6)  # Remove invalid entries

# numba based
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

def main():
    X0X1, Ns = generate_dataset()
    X0X1 = np.array(X0X1)
    Ns = np.array(Ns)

    r1 = chop_np(X0X1, Ns)
    r2 = chop_nb(X0X1, Ns)
    r3 = chop_np2(X0X1, Ns)
    r4 = chop_np_optimized(X0X1, Ns)

    r1 = chop_np(X0X1, Ns)
    r2 = chop_nb(X0X1, Ns)
    r3 = chop_np2(X0X1, Ns)
    r4 = chop_np_optimized(X0X1, Ns)

    r1 = chop_np(X0X1, Ns)
    r2 = chop_nb(X0X1, Ns)
    r3 = chop_np2(X0X1, Ns)
    r4 = chop_np_optimized(X0X1, Ns)

    assert np.allclose(r1, r2)
    assert np.allclose(r1, r3)
    assert np.allclose(r1, r4)

    t1 = timeit.Timer(stmt='chop_np(X0X1, Ns)', setup='from __main__ import chop_np', globals={'X0X1' : X0X1, 'Ns' : Ns})
    t2 = timeit.Timer(stmt='chop_nb(X0X1, Ns)', setup='from __main__ import chop_nb', globals={'X0X1' : X0X1, 'Ns' : Ns})
    t3 = timeit.Timer(stmt='chop_np2(X0X1, Ns)', setup='from __main__ import chop_np2', globals={'X0X1' : X0X1, 'Ns' : Ns})
    t4 = timeit.Timer(stmt='chop_np_optimized(X0X1, Ns)', setup='from __main__ import chop_np_optimized', globals={'X0X1' : X0X1, 'Ns' : Ns})
    nloop1, time1 = t1.autorange()
    nloop2, time2 = t2.autorange()
    nloop3, time3 = t3.autorange()
    nloop4, time4 = t4.autorange()
    print('np', time1/nloop1 * 1E3,'ms for', nloop1, 'runs')
    print('nb', time2/nloop2 * 1E3,'ms for', nloop2, 'runs')
    print('np2', time3/nloop3 * 1E3,'ms for', nloop3, 'runs')
    print('np_optimized', time4/nloop4 * 1E3,'ms for', nloop4, 'runs')

if __name__ == '__main__':
    main()
