# G4Step division in a vectorized way
## Algorithm
### Input and output
Input data is an array of steps, in a size of (N, 6).
A step, (1,6), will be divided to smaller pieces when its length larger than a threshold.
The new steps have a length smaller than or equal to threshold.
It is not guaranteed that each original step yields the same number of divisions given the threshold.
### Algorithm description
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
  
We observe:
- The vectorized way of `chop_np_optimized` has a performance close to compiled version. This is consistent with expectation.
  The vectorized way needs additional calculations and indexing but for-loop is in absence.
- The version with pre-allocated array is faster than the one in which a lot of temporary arrays are created.
  - This is opposite to torch, where the index access has bigger effects.

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

        def idx(X):
            for i in range(len(X)):
                X[i,0]
            return
1. Test data

        X = torch.rand((1000,2))
        Y = np.random.rand(2000).reshape(1000, 2)
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

### for-loop based indexing for two input
- Test function

        def idx(X, Y):
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

### CUDA synchronization in for-loop.
- Test function

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

- Test data

        X = torch.rand((1000, 2))

- Test platform
  - CPU: AMD Ryzen Threadripper 7970X 32-Cores
  - GeForce RTX 4090
- Result
  - %timeit idx1: 797 μs ± 704 ns per loop
  - %timeit idx2: 6.61 ms ± 18 μs per loop
  - %timeit idx3: 7.03 μs ± 21.7 ns per loop
  - %timeit idx4: 1.36 μs ± 4.24 ns per loop
- Observation
  - Function `idx1` yields similar time as function `idx` without explicit synchronization.
  - Function '`idx4` is significantly faster than `idx3` as %timeit does not wait for cuda synchronization.
    - We should do `%timeit torch.cuda.synchronize(); idx4(X);
torch.cuda.synchronize()`, which yields 7.37 μs ± 7.67 ns per loop.
- Lesson
  - Torch automatically handle synchronization at the start and the
    end. Manual synchronization may take extra time.

- Reference
  - CUDA operations are asynchronous w.r.t host.
    - https://pytorch.org/docs/main/notes/cuda.html
    - https://discuss.pytorch.org/t/is-it-possible-to-run-two-functions-in-parrallel-under-the-eager-mode-during-training/191296/2

### Advance indexing
- Test function

        def idx1tolast(X):
            return X[1:]
- Test data

        X = torch.rand((1000_000,))
        X = torch.rand((10,))
        X = torch.rand((5,))
        Y = np.random.rand(10)

- Test platform
  - CPU: AMD Ryzen Threadripper 7970X 32-Cores
  - GeForce RTX 4090
- Result
  - numpy, input (10, ): 71.4 ns ± 0.177 ns per loop
  - numpy, input (5, )  70 ns ± 0.138 ns per loop
  - torch on CPU, input (10, ): 806 ns ± 0.432 ns per loop
  - torch on CPU, input (5,): 829 ns ± 0.421 ns per loop
  - torch on CPU, input (1000,): 800 ns ± 0.662 ns per loop
  - torch on GPU, input (10,): 6.98 μs ± 11.3 ns per loop
  - torch on GPU, input (1000_000,): 6.95 μs ± 9.73 ns per loop
- Observation
  - Advance indexing in numpy is faster than torch. Torch performs better on CPU than GPU for a small dataset.

### Factory method, linspace
- Test function

        def np_linspace(x0, x1, n):
            return np.linspace(x0, x1, n)
        def torch_linspace(x0, x1, n, device='cpu'):
            return torch.linspace(x0, x1, n, requires_grad=False, device=device)
- Test data

        x0 = 20
        x1 = 30
- Test platform
  - CPU: AMD Ryzen Threadripper 7970X 32-Cores
  - GeForce RTX 4090
- Result
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
- Observation
  - Torch version on CPU is faster than on numpy on GPU by a factor of 2--3.
  - Torch on GPU shows constantly behavior until the output size is large enough.
  - Numba is surpassed by torch on GPU at a very large size.

### Assemble for-loop, indexing, slicing, linspace:
1. Test function:

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
  - Source files are [here](simple_test.py) and [there](simple_test3.py).
1. Expectation:
  - The input is in a size (1000, 6) and (1000,).
  - For-loop based indexing appears 9 times.
    - numpy: 60us/1000 * 10_000 * 9 ~ 6ms
      - numpy performance does scale with 9 but some number smaller than 9.
    - torch on CPU: 1.3ms/1000 * 10_000 * 9 ~ 120 ms
    - torch on GPU: 7us * 10_000 * 9 ~ 630 ms; we use time for
      `synchronize; X[i,0]; synchronize` as `linspace` need inputs
      from synchronized elements.
  - In each iteration, `linspace` appears 3 times.
    - numpy: 5us * 10_000 * 3 ~ 150 ms
    - torch on CPU: 1.5us *10_000 * 3 ~ 50ms
    - torch on GPU: 5us * 10_000 * 3 ~ 150ms
  - In each itertion, slicing appears 6 times.
    - numpy: 0.08us * 10_000 * 6 ~ 5ms
    - torch on CPU: 0.8us * 10_000 * 6 ~ 50 ms
    - torch on GPU: 1u * 10_000 * 6 ~ 60ms
  - In total:
    - numpy: O(160ms)
    - torch on CPU: O(220ms)
    - torch on GPU: O(840ms)
1. Test data

        from common_dataset import generate_dataset
        X0X1, Ns= generate_dataset()
        # second data
        X = X0X1[:1000]
        N = Ns[:1000]
1. Results
   - numpy, input (10_000,): 198 ms
   - torch on CPU, input (10_000,): 282 ms
   - torch on GPU, input (10_000,): 800 ms
   - numpy, input (1_000,): 20 ms
   - torch on CPU, input (1_000,): 29 ms
1. Lessons
   - Expressions on CUDA need time to synchronize. When we estimates the runtime, we cannot use the time for which synchronization are done until loop finishes.
1. One more test
   - Test script [simple_test4.py](simple_test4.py) demonstration the influence of synchronization, by `m3` and `m4`.

### Stack things
- Test function

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

- Results
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

- Observation
  - Torch on CPU and on GPU handles better on large data, while numpy
    and numba handles small data better.
- Lesson
  - The method np.stack in numpy is NO better than assigning values to preallocated array.

### Assemble for-loop, indexing, slicing, linspace, stack
- Test function
  - Source is [here](simple_test2.py)
- Test data

        from common_dataset import generate_dataset
        X0X1, Ns= generate_dataset()
- Test platform
  - CPU
  - GPU
- Expectation
  - Stack for numpy: 30ms
  - Stack for torch on CPU: 25ms
  - Total for numpy: O(200ms)
  - Total for torch on CPU: (250ms)
- Results
  - numpy: 240ms
  - torch on CPU: 330ms
## Function `torch.vmap` does not work as of today.
We attempt to define a function for each step and vectorize it using
[torch.vmap](https://pytorch.org/docs/stable/generated/torch.vmap.html) along batch dimension:
```
def chop_seg(seg, n):
    idxs = torch.arange(n+1) / n
    xs = seg[0] * (1-idxs) + seg[3] * idxs
    ys = seg[1] * (1-idxs) + seg[4] * idxs
    zs = seg[2] * (1-idxs) + seg[5] * idxs
    return torch.stack([xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]], dim=1).view(-1, 6)
chop = torch.vmap(chop)
```
or
```
def chop_seg(seg, n):
    xs = torch.linspace(seg[0], seg[3], n)
    ys = torch.linspace(seg[1], seg[4], n)
    zs = torch.linspace(seg[2], seg[5], n)
    return torch.stack([xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]], dim=1).view(-1, 6)
chop = torch.vmap(chop)
```
Unfortunately, function [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html)
and [torch.linspac](https://pytorch.org/docs/stable/generated/torch.linspace.html)e require [scalar](https://pytorch.org/cppdocs/notes/tensor_basics.html)
inputs, rather than a tensor, for some arguments.
Method `torch.vmap` does not support scalar. We encounter the problem as of today.
```
RuntimeError: vmap: It looks like you're calling .item() on a Tensor. We don't support vmap over calling .item() on a Tensor, please try to rewrite what you're doing with other operations. If error is occurring somewhere inside PyTorch internals, please file a bug report.
```
A [GitHub issue](https://github.com/pytorch/pytorch/issues/105494) is linked here.

A another try is to have `idxs = torch.torch(tuple(i for i in range(n+1))) / n`.
But it will return `RuntimeError: all inputs of range must be ints, found Tensor in argument 0`.

### Does preallocated array helps? No for torch...
We already see

## torch and JIT
### The active project: torch.compile
The library
[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
is is intended to replace
[TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).
It records the graph of operations and compile the graph. I copied
information from the
[blog](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/).
> PyTorch supports two execution modes [1]: eager mode and graph
> mode. In eager mode, operators in a model are immediately executed
> as they are encountered. In contrast, in graph mode, operators are
> first synthesized into a graph, > which will then be compiled and
> executed as a whole.

The decorator `torch.compile` can handle the data-dependent flow.  The
price is to have 'graph breaks' in the code. Each condition under an
input leads to a recompilation and graph. An example is
[here](https://discuss.pytorch.org/t/compiling-for-loop-makes-it-run-25x-slower-than-uncompiled-version/191488).

The non-pytorch type, such as python integers, are treated as
constants and can trigger re-compilations as long as they change.

An example is in the printed message from [torch_chop_compile.py](torch_chop_compile.py)
> Elapsed time for X0X1[:8_000] 678.5413208007812 milliseconds using torch_chop_optimized_compile

To check if there are re-compilations, try
```
TORCH_LOGS='recompiles' python script.py
```
It is also possible to force to compile a full graph. An exception will be thrown when 'graph breaks' present.
```
compiled_fn = torch.compile(fn, fullgraph=True)
```

Read more in [FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html).

### The legacy way: TorchScript
There are two ways to compile a graph in
[TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).
- trace: it trace the graph and is more friendly to python code but it
  will give wrong results if data-dependent control flow presents.
- script: it compile the graph using static data type and is able to
  handle control flow. But it only works with a subset of python
  features.
  
#### for-loop is unrolled
See [page](torch_cjit_chop.py).

Tests are available in [torch_chop_compile2.py](torch_chop_compile2.py) and [torch_chop_compile3.py](torch_chop_compile3.py).
Test data are from `common_dataset.py`.
Test functions and results are:
- `torch_chop`: results are pre-allocated and in-place assignments are
  done per iteration. In-place assignments are done in for-loop.
  - CPU: Median: 453.27 ms
  - GPU: 2.02s
- `torch_chop`: with unrolled loop:
  - CPU: Median: 442.06 ms
  - GPU Median: 2s
- `torch_chop2`: results are pre-allocated and in-place assignments
  are done per iteration. CUDA sync are also called per
  iteration. In-place assignments are done in for-loop.
- `torch_chop_compile`: by torch.compile
  - CPU: Median 452.18 ms
  - GPU: 2.03s
- `torch_chop_compile` in `torch_chop_compile3.py`, with unrolled loop
  - CPU: Median: 439.55 ms
  - GPU: 2s
- `torch_chop_script`: by torch.jit.script
  - CPU: Median 344.43 ms
  - GPU: 1.44s
- `torch_chop_script`: in `torch_chop_compile3.py`, with unrolled loop
  - CPU: Median: 345.57 ms
  - GPU: 1.46s
- `torch_chop2_script`: by torch.jit.script
  - CPU: Median 351.37 ms
  - GPU: 1.44s
  
### Custom C++ extensions and CUDA kernels
Read more in [page](https://pytorch.org/tutorials/advanced/cpp_extension.html).
The performance is not fully optimized without writing CUDA kernels...
See tests in [torch_cjit_chop.py](torch_cjit_chop.py).

### Concerns in this problem
In the for-loop based solution, we have data-dependent control flow, as `torch.linspace(X[i,0], X[i,3], Ns[i])`.
### A few tests
#### torch.compile, torch.jit.script, eager mode, and numpy/numba
- Test data
  - in `common_dataset.py`.
- Test function and results
  - Functions are in [torch_chop_compile.py](torch_chop_compile.py),
    and [npnb_jit_chop.py](npnb_jit_chop.py), and
    [torch_cjit_chop.py](torch_cjit_chop.py).
  - `torch_chop`: optimized version in eager-mode using for-loop, torch.stack, torch.cat.
    - on CPU: Median: 291.33 ms
    - on GPU: Median: 947.02 ms
  - `torch_chop_compile`: by `torch.compile(torch_chop)`.
    - on CPU: Median: 293.28 ms
    - on GPU: Median: 940.46 ms
  - `torch_chop_script`: by `torch.jit.script(torch_chop)`.
    - on CPU: Median: 130.87 ms
    - on GPU: Median: 575.25 ms
  - `torch_chop_optimized`: optimized version in a vectorized way.
    - on CPU: Median: 815.35 us
    - on GPU: Median: 182.52 us
  - `torch_chop_optimized_compile`: by `torch.compile`.
    - on CPU: Median: 528.48 us; without recompilations
    - on GPU: Median: 143.58 us; without recompilations
  - `torch_chop_optimized_script`: by `torch.jit.script`.
    - on CPU: Median: 766.22 us
    - on GPU: Median: 135.42 us
  - `batch_chop_X0X1`: by C++ front-end, using for-loop, advance indexing and in-place manipulations
    - on CPU: Median: 192.89 ms
    - on GPU: Median: 1.17 s
  - `batch_chop_X0X1_v2`: by C++ front-end, using for-loop, torch.stack, torch.cat.
    - on CPU: 131.86 ms
    - on GPU: Median: 624.88 ms
    - Performance is very close to `torch_chop_script`.
