# G4Step division in a vectorized way
## Algorithm
### Input and output
Input data is an array of steps, in a size of (N, 6).
A step, (1,6), will be divided to smaller pieces when its length larger than a threshold.
The new steps have a length smaller than or equal to threshold.
It is not guaranteed that each original step yields the same number of divisions given the threshold.
### Algorithm
- Compute the length of each step
- Compute the number of division for each step, given a universal threshold.
  - Steps shorter than the threshold are kept.
  - Steps longer than the threshold are chopped.
- Chop each step equally according to the number of steps.
  - `linspace` rather than `arange` as we do not want to deal with left over of the last division.

The first two items are easy to achieve in a vectorized way.
```
LdX = np.linalg.norm(X0-X1, axis=1)
Ns = (LdX // threshold).astype(np.int32) + 1 # + 1 as we want to limit the step size smaller than threshold
for i in range(len(Ns)):
    np.linspace(start,end, Ns[i]+1) # +1 because this is an array containing both start and end points.
```

It is not an easy work to vectorize for-loop. One possible try is:
```
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
```
See files for [numpy](npnb_jit_chop.py) and [torch](torch_chop_compile.py).

## Numba and numpy
numba has supports to numpy, including
- [numpy.linspace](https://numba.pydata.org/numba-doc/dev/developer/autogen_numpy_listing.html?highlight=linspace#numpy.core.function_base.linspace)
- [efficient index accessing](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#:~:text=NumPy%20arrays%20are%20directly%20supported%20in%20Numba.%20Access%20to%20Numpy%20arrays%20is%20very%20efficient%2C%20as%20indexing%20is%20lowered%20to%20direct%20memory%20accesses%20when%20possible.)

Tests are performed in [npnb_jit_chop.py](npnb_jit_chop.py).
- Function `chop_np`: 225.37532029673457 ms for 1 runs.
  - It consists of a for-loop and assigning new steps to a preloaded array.
- Function `chop_nb`: 0.7169480174779892 ms for 500 runs.
  - It is a compiled version of `chop_np`.
- Function `chop_np2`: 237.9163159057498 ms for 1 run
  - It consists of a for-loop and concatenating the temporary arrays at the last step.
- Function `chop_np_optimized`: 0.9392065620049834 ms for 500 runs
  - It is the vectorized version without for-loop.
The vectorized way of `chop_np_optimized` has a performance close to compiled version. This is consistent with expectation.
The vectorized way needs additional calculations and indexing but for-loop is in absence.

## Numpy and torch
### for-loop based indexing for one input
- Test function
        def idx(X):
            for i in range(len(X)):
                X[i]
            return
- Test data
        X = torch.rand((1000,2))
        X = torch.rand((1_000, 6))
        Y = np.random.rand(2000).reshape(1000, 2)
- Test platform
  - CPU: AMD Ryzen Threadripper 7970X 32-Cores
  - GeForce RTX 4090.
- Result
  - numba: 103 ns ± 0.273 ns per loop
  - numpy: 55.3 μs ± 249 ns per loop
  - torch on CPU, input (1000, 2): 635 μs ± 1.17 μs per loop
  - torch on CPU, input (1000, 6): 676 μs ± 412 ns per loop
  - torch on GPU: 779 μs ± 559 ns per loop
- Observation
  - JITed version by numba is the fastest. Numpy indexing is faster
    than torch indexing on the first axis. Torch on GPU is the
    slowest.

### for-loop based indexing for two inputs
- Test function
        def idx(X):
            for i in range(len(X)):
                X[i]
                Y[i]
            return
- Test data
        X = torch.rand((1000, 2))
    Y = torch.rand((1000, ))
- Test platform
  - CPU: AMD Ryzen Threadripper 7970X 32-Cores
  - GeForce RTX 4090
- Result
  - numba: 130 ns ± 0.175 ns
  - numpy: 79.5 μs ± 667 ns per loop
  - torch on CPU: 1.34 ms ± 1.42 μs per loop
- Observation
  - Runtime increases for both numpy and torch. Runtime become double for torch.

### for-loop based indexing for 2D input
1. Test function
  ```python
        def idx(X):
            for i in range(len(X)):
                X[i,0]
            return
  ```
1. Test data
  ```python
  X = torch.rand((1000,2))
  Y = np.random.rand(2000).reshape(1000, 2)
  ```
- Test platform
  - CPU: AMD Ryzen Threadripper 7970X 32-Cores
  - GeForce RTX 4090.
- Result
  - numba: 102 ns ± 0.418 ns per loop
  - numpy: 53.7 μs ± 667 ns per loop
  - torch on CPU: 1.3 ms ± 367 ns per loop
  - torch on GPU: 1.46 ms ± 3.82 μs per loop
- Observation
  - Runtime for numpy does not vary too much. Runtime for torch become double.

=== for-loop based indexing for two input
==== Test function
```
def idx(X, Y):
    for i in range(len(X)):
        X[i]
        Y[i]
    return
```
==== Test data
```
X = torch.rand((1000, 2))
Y = torch.rand((1000, ))
```
==== Test platform
- CPU: AMD Ryzen Threadripper 7970X 32-Cores
- GeForce RTX 4090
==== Result
- numba: 130 ns ± 0.175 ns
- numpy: 79.5 μs ± 667 ns per loop
- torch on CPU: 1.34 ms ± 1.42 μs per loop
==== Observation
Runtime increases for both numpy and torch. Runtime become double for torch.

=== CUDA synchronization in for-loop.
==== Test function
```
def idx1(X):
    torch.cuda.synchronize()
    for i in range(len(X)):
        y = X[i]
    torch.cuda.synchronize()

def idx2(X):
    for i in range(len(X)):
        torch.cuda.synchronize()
        y = X[i]
        torch.cuda.synchronize()
        
def idx3(X):
    torch.cuda.synchronize()
    X[0,0]
    torch.cuda.synchronize()
def idx4(X):
    X[0,0]
```
==== Test data
```
X = torch.rand((1000, 2))
```
==== Test platform
- CPU: AMD Ryzen Threadripper 7970X 32-Cores
- GeForce RTX 4090
==== Result
- %timeit idx1: 797 μs ± 704 ns per loop
- %timeit idx2: 6.61 ms ± 18 μs per loop
- %timeit idx3: 7.03 μs ± 21.7 ns per loop
- %timeit idx4: 1.36 μs ± 4.24 ns per loop
==== Observation
Function `idx1` yields similar time as function `idx` without explicit synchronization.
Function 'idx4' is significantly faster than `idx3` as 
==== Lesson
Torch automatically handle synchronization at the start and the end. Mannual syncrhonization wastes time.
The longer time of indexing on GPU than CPU have nothing to do syncrhonization problem.
==== Reference
CUDA operations are asynchronous. https://pytorch.org/docs/main/notes/cuda.html

=== Advance indexing
==== Test function
```
def idx1tolast(X):
    return X[1:]
```
==== Test data
```
X = torch.rand((10,))
X = torch.rand((5,))
Y = np.random.rand(10)
```
==== Test platform
- CPU: AMD Ryzen Threadripper 7970X 32-Cores
- GeForce RTX 4090
==== Result
- numpy, input (10, ): 71.4 ns ± 0.177 ns per loop
- numpy, input (5, )  70 ns ± 0.138 ns per loop
- torch on CPU, input (10, ): 784 ns ± 1.45 ns per loop
- torch on CPU, input (5,): 762 ns ± 0.421 ns per loop
- torch on GPU: 924 ns ± 0.501 ns per loop
==== Observation
Advance indexing in numpy is faster than torch. Torch performs better on CPU than GPU.

=== Factory method, linspace
==== Test function
```
def np_linspace(x0, x1, n):
    return np.linspace(x0, x1, n)
def torch_linspace(x0, x1, n, device='cpu'):
    return torch.linspace(x0, x1, n, requires_grad=False, device=device)
```
==== Test data
N/A.
==== Test platform
- CPU: AMD Ryzen Threadripper 7970X 32-Cores
- GeForce RTX 4090
==== Result
- `linspace(20, 30, 1_000_000)`
  - torch on GPU: 8.7 μs ± 1.01 ns per loop
- `linspace(20, 30, 100_000)`
  - numba: 6.42 μs ± 7.11 ns per loop
  - torch on GPU: 3.66 μs ± 1.24 ns per loop
- `linspace(20, 30, 10_000)`
  - numba: 1.01 μs ± 2.3 ns per loop
  - numpy: 10.5 μs ± 4.36 ns per loop
  - torch on CPU: 5.96 μs ± 5.32 ns per loop
  - torch on GPU: 3.64 μs ± 1.64 ns per loop
- `linspace(20, 30, 30)`
  - numpy: 4.97 μs ± 4.12 ns per loop
  - numba: 424 ns ± 0.833 ns per loop
  - torch on CPU: 1.6 μs ± 4.2 ns per loop
  - torch on GPU: 3.74 μs ± 3.44 ns per loop
- `linspace(20, 30, 20)`
  - numpy: 4.95 μs ± 3.54 ns per loop
  - numba: 410 ns ± 1.48 ns per loop
  - torch on CPU: 1.57 μs ± 2.09 ns per loop
  - torch on GPU: 3.73 μs ± 2.77 ns per loop
- `linspace(20, 30, 2)`
  - numpy: 4.82 μs ± 4.58 ns per loop
  - numba: 402 ns ± 0.414 ns per loop
  - torch on CPU: 1.56 μs ± 2.23 ns per loop
  - torch on GPU: 3.74 μs ± 4.16 ns per loop
==== Observation
Torch version on CPU is faster than on numpy on GPU by a factor of 2--3.
Torch on GPU shows constantly behavior until the output size is large enough.
Numba is surpassed by torch on GPU at a very large size.

=== Assemble for-loop, indexing, slicing, linspace:
==== Test function:
```
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
        xs = torch.linspace(X[i,0], X[i,3], Ns[i]+1)
        ys = torch.linspace(X[i,1], X[i,4], Ns[i]+1)
        zs = torch.linspace(X[i,2], X[i,5], Ns[i]+1)
        xs[:-1]
        ys[:-1]
        zs[:-1]
        xs[1:]
        ys[1:]
        zs[1:]
```
Source file are [here](simple_test.py) and [there](simple_test3.py).
==== Expectation:
The input is in a size (1000, 6) and (1000,).

For-loop based indexing appears 9 times.
- numpy: 60us/1000 * 10_000 * 9 ~ 6ms
  - numpy performance does scale with 9 but some number smaller than 9.
- torch on CPU: 1.3ms/1000 * 10_000 * 9 ~ 120 ms
- torch on GPU: 7us * 10_000 * 9 ~ 630 ms; we use time for
  `synchronize; X[i,0]; synchronize` as `linspace` need inputs from
  synchronized elements.

In each iteration, `linspace` appears 3 times.
- numpy: 5us * 10_000 * 3 ~ 150 ms
- torch on CPU: 1.5us *10_000 * 3 ~ 50ms
- torch on GPU: 5us * 10_000 * 3 ~ 150ms

In each itertion, slicing appears 6 times.
- numpy: 0.08us * 10_000 * 6 ~ 5ms
- torch on CPU: 0.8us * 10_000 * 6 ~ 50 ms
- torch on GPU: 1u * 10_000 * 6 ~ 60ms

In total:
- numpy: O(160ms)
- torch on CPU: O(220ms)
- torch on GPU: O(840ms)
==== Test data
```
from common_dataset import generate_dataset
X0X1, Ns= generate_dataset()
# second test data
X = X0X1[:1000]
N = Ns[:1000]
```
==== Results:
numpy, input (10_000,): 198 ms
torch on CPU, input (10_000,): 282 ms
torch on GPU, input (10_000,): 800 ms
numpy, input (1_000,): 20 ms
torch on CPU, input (1_000,): 29 ms
==== Lessons
Expressions on CUDA need time to synchronize. When we estimates the runtime, we cannot use the time for which synchronization are done until loop finishes.
==== One more tests
Test script [simple_test4.py](simple_test4.py) demonstration the influence of synchronization, by `m3` and `m4`.

==== Stack things
==== Test function
```
def torch_stack(X):
    return torch.stack(X).view(-1,6)

def np_stack(X):
    return np.stack(X, axis=1).reshape(-1, 6)

# numba does not support Python List
@nb.njit
def nb_stack(x0, x1, x2, x3, x4, x5):
    return np.stack((x0, x1, x2, x3, x4, x5), axis=1).reshape(-1, 6)

@nb.njit
def nb_stack(x0, x1, x2, x3, x4, x5):
    result = np.empty((len(x0), 6))
    result[:,0] = x0
    result[:,1] = x1
    result[:,2] = x2
    result[:,3] = x3
    result[:,4] = x4
    result[:,5] = x5
    return result
```
==== Results
- numba, input 6x(3,) : 738 ns ± 1.15 ns per loop
- numba, input 6x(10_000,) : 54.8 μs ± 7.28 ns per loop
- numba, manual copy, input 6x(3,): 646 ns ± 0.95 ns per loop
- numba, manual copy, input 6x(10_000, ): 24.6 μs ± 19.9 ns per loop
- numpy, input 6x(3,): 2.98 μs ± 4.78 ns per loop
- numpy, input 6x(10_000,): 26.8 μs ± 30.9 ns per loop
- torch on CPU, input 6x(3,): 2.24 μs ± 3.13 ns per loop
- torch on GPU, input 6x(3,): 5.2 μs ± 1.39 ns per loop
- torch on CPU, input 6x(10_000,): 7.4 μs ± 2.15 ns per loop
- torch on GPU, input 6x(10_000,): 5.23 μs ± 5 ns per loop
- torch on GPU, input 6x(1_000_000,): 14.3 μs ± 8.96 ns per loop

==== Observation
Torch on CPU and on GPU handles better on large data, while numpy and numba handles small data better.
==== Lesson
The method np.stack in numpy is no better than assigning values to preallocated array.

=== Assemble for-loop, indexing, slicing, linspace, stack
==== Test function
Source is [here](simple_test2.py)
==== Test data
```
from common_dataset import generate_dataset
X0X1, Ns= generate_dataset()
```
==== Test platform

==== Expection
Stack for numpy: 30ms
Stack for torch on CPU: 25ms
Total for numpy: O(200ms)
Total for torch on CPU: (250ms)

==== Results
- numpy: 240ms
- torch on CPU: 330ms
