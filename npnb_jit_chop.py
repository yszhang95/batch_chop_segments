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

    r1 = chop_np(X0X1, Ns)
    r2 = chop_nb(X0X1, Ns)
    r3 = chop_np2(X0X1, Ns)

    r1 = chop_np(X0X1, Ns)
    r2 = chop_nb(X0X1, Ns)
    r3 = chop_np2(X0X1, Ns)

    assert np.allclose(r1, r2)
    assert np.allclose(r1, r3)
    t1 = timeit.Timer(stmt='chop_np(X0X1, Ns)', setup='from __main__ import chop_np', globals={'X0X1' : X0X1, 'Ns' : Ns})
    t2 = timeit.Timer(stmt='chop_nb(X0X1, Ns)', setup='from __main__ import chop_nb', globals={'X0X1' : X0X1, 'Ns' : Ns})
    t3 = timeit.Timer(stmt='chop_np2(X0X1, Ns)', setup='from __main__ import chop_np2', globals={'X0X1' : X0X1, 'Ns' : Ns})
    nloop1, time1 = t1.autorange()
    nloop2, time2 = t2.autorange()
    nloop3, time3 = t3.autorange()
    print('np', time1/nloop1 * 1E3,'ms')
    print('nb', time2/nloop2 * 1E3,'ms')
    print('np2', time3/nloop3 * 1E3,'ms')

if __name__ == '__main__':
    main()
