import torch
import torch.utils.benchmark as benchmark
from common_dataset import generate_dataset

# torch based
def chop_torch(X0X1, Ns):
    output = []
    for i in range(len(Ns)):
        size = Ns[i]
        xs = torch.linspace(X0X1[i,0], X0X1[i,3], size+1, device=Ns.device)
        ys = torch.linspace(X0X1[i,1], X0X1[i,4], size+1, device=Ns.device)
        zs = torch.linspace(X0X1[i,2], X0X1[i,5], size+1, device=Ns.device)
        output.append(torch.stack([xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]], dim=1).view(-1, 6))
    return torch.cat(output, dim=0)

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

def chop_torch_optimized_debug(X0X1, Ns):
    batch_size = X0X1.shape[0]
    device = X0X1.device

    # Find the max segment count to unify tensor sizes
    max_size = Ns.max().item()
    N_LIMIT = 1000
    MIN_STEP = 1/N_LIMIT
    assert max_size + 1 < N_LIMIT

    # Debug: Print input shapes
    print(f"X0X1.shape: {X0X1.shape}, Ns.shape: {Ns.shape}")
    # Create a common index tensor (0 to max_size)
    idxs = torch.arange(max_size + 0.1, device=device).float()  # Shape: (max_size + 1,)
    # Debug: Print index shape
    print(f"idxs.shape: {idxs.shape}")

    # Normalize indices based on Ns (broadcasting)
    frac_steps = idxs[:, None] / Ns[None, :]  # Shape: (max_size+1, batch_size)
    frac_steps = frac_steps.T  # Shape: (batch_size, max_size+1)
    # Debug: Print frac_steps shape
    print(f"frac_steps.shape: {frac_steps.shape}")

    # Mask to ignore extra indices for each row
    mask = frac_steps < (1+MIN_STEP)

    # Compute interpolated values using broadcasting
    xs = (1 - frac_steps) * X0X1[:, 0, None] + frac_steps * X0X1[:, 3, None]
    ys = (1 - frac_steps) * X0X1[:, 1, None] + frac_steps * X0X1[:, 4, None]
    zs = (1 - frac_steps) * X0X1[:, 2, None] + frac_steps * X0X1[:, 5, None]

    # Debug: Print interpolated values shape
    print(f"xs.shape: {xs.shape}, ys.shape: {ys.shape}, zs.shape: {zs.shape}")

    # Apply mask
    xs, ys, zs = xs * mask, ys * mask, zs * mask

    mask = frac_steps < (1-MIN_STEP) # discard the last point as the lenght of output is less than the lenght of linspace by 1.

    # Stack and reshape for output format
    output = torch.stack([xs[:, :-1], ys[:, :-1], zs[:, :-1], xs[:, 1:], ys[:, 1:], zs[:, 1:]], dim=-1)
    return output[mask[:, :-1]].view(-1, 6)  # Remove invalid entries

chop_torch_compile = torch.compile(chop_torch)
# chop_torch_script = torch.jit.script(chop_torch, fullgraph=True) # Throw an exception when there exists a data-dependent control flow; for example, size=Ns[i] for linspace(x0, x1, size)
chop_torch_script = torch.jit.script(chop_torch)
chop_torch_optimized_compile= torch.compile(chop_torch_optimized)
chop_torch_optimized_script = torch.jit.script(chop_torch_optimized)
chop_torch_optimized_debug_script = torch.jit.script(chop_torch_optimized_debug)

def test_device(device):
    print('\n------------------------------------')
    print('Testing device', device)

    X0X1, Ns = generate_dataset()
    X0X1new = X0X1.to(device)
    Nsnew = Ns.to(device)

    r1 = chop_torch_compile(X0X1new, Nsnew)
    r2 = chop_torch_script(X0X1new, Nsnew)
    r3 = chop_torch(X0X1new, Nsnew)
    r4 = chop_torch_optimized(X0X1new, Nsnew)
    r5 = chop_torch_optimized_compile(X0X1new, Nsnew)
    r6 = chop_torch_optimized_script(X0X1new, Nsnew)

    print('r1: chop_compile is on', r1.device)
    print('r2: chop_script is on', r2.device)
    print('r3: python is on', r3.device)
    print('r4: optimized python is on', r4.device)
    print('r5: optimized compile is on', r5.device)
    print('r6: optimized script is on', r6.device)

    assert r1.allclose(r3)
    assert r2.allclose(r3)
    assert r4.allclose(r3)
    assert r5.allclose(r3)
    assert r6.allclose(r3)

    t1 = benchmark.Timer(
        stmt = 'chop_torch_compile(X0X1, Ns)',
        setup = 'from __main__ import chop_torch_compile',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    t2 = benchmark.Timer(
        stmt = 'chop_torch_script(X0X1, Ns)',
        setup = 'from __main__ import chop_torch_script',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    t3 = benchmark.Timer(
        stmt = 'chop_torch(X0X1, Ns)',
        setup = 'from __main__ import chop_torch',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    t4 = benchmark.Timer(
        stmt = 'chop_torch_optimized(X0X1, Ns)',
        setup = 'from __main__ import chop_torch_optimized',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    t5 = benchmark.Timer(
        stmt = 'chop_torch_optimized_compile(X0X1, Ns)',
        setup = 'from __main__ import chop_torch_optimized_compile',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    t6 = benchmark.Timer(
        stmt = 'chop_torch_optimized_script(X0X1, Ns)',
        setup = 'from __main__ import chop_torch_optimized_script',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    for i in range(3):
        chop_torch_compile(X0X1new, Nsnew)
    print(t1.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch_script(X0X1new, Nsnew)
    print(t2.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch(X0X1new, Nsnew)
    print(t3.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch_optimized(X0X1new, Nsnew)
    print(t4.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch_optimized_compile(X0X1new, Nsnew)
    print(t5.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch_optimized_script(X0X1new, Nsnew)
    print(t6.blocked_autorange(min_run_time=1))
    try:
        X0X1new = X0X1new[0:8_000]
        Nsnew = Nsnew[0:8_000]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        chop_torch_optimized_compile(X0X1new, Nsnew)
        end.record()
        torch.cuda.synchronize()
        print('Elapsed time for X0X1[:8_000]', start.elapsed_time(end), 'milliseconds using chop_torch_optimized_compile')
    except Exception as e:
        print('Failed to rerun chop_torch_optimized_compile for X0X1[:8_000] on', X0X1new.device)
        print(e)
    try:
        X0X1new = X0X1new[0:8_000]
        Nsnew = Nsnew[0:8_000]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        chop_torch_optimized_script(X0X1new, Nsnew)
        end.record()
        torch.cuda.synchronize()
        print('Elapsed time for X0X1[:8_000]', start.elapsed_time(end), 'milliseconds using chop_torch_optimized_script')
    except Exception as e:
        print('Failed to rerun chop_torch_optimized_script for X0X1[:8_000] on', X0X1new.device)
        print(e)
    print(chop_torch_script.graph_for(X0X1new, Nsnew))

if __name__ == '__main__':
    test_device(device='cpu')
    test_device(device='cuda:0')
