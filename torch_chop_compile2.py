import torch
import torch.utils.benchmark as benchmark
from common_dataset import generate_dataset

# torch based
def chop_torch(X0X1, Ns, result):
    idx = 0
    for i in range(len(Ns)):
        size = Ns[i]
        for j in range(3):
            temp = torch.linspace(X0X1[i,j], X0X1[i,j+3], size+1, device=Ns.device)
            result[idx:idx+size,j] = temp[:-1]
            result[idx:idx+size,j+3] = temp[1:]
        idx += size  # Move the index forward
    return result

def chop_torch2(X0X1, Ns, result):
    idx = 0
    for i in range(len(Ns)):
        torch.cuda.synchronize()
        size = Ns[i]
        for j in range(3):
            temp = torch.linspace(X0X1[i,j], X0X1[i,j+3], size+1, device=Ns.device)
            result[idx:idx+size,j] = temp[:-1]
            result[idx:idx+size,j+3] = temp[1:]
        idx += size  # Move the index forward
        torch.cuda.synchronize()
    return result

chop_torch_compile = torch.compile(chop_torch)
chop_torch_script = torch.jit.script(chop_torch)
chop_torch_script2 = torch.jit.script(chop_torch2)

def test_device(device):
    print('Testing device', device)

    X0X1, Ns = generate_dataset()
    X0X1new = X0X1.to(device)
    Nsnew = Ns.to(device)

    results = torch.empty((torch.sum(Nsnew), 6), device=device, requires_grad=False)

    r1 = chop_torch_compile(X0X1new, Nsnew, results)
    r2 = chop_torch_script(X0X1new, Nsnew, results)
    r3 = chop_torch(X0X1new, Nsnew, results)
    r4 = chop_torch_script2(X0X1new, Nsnew, results)

    print('r1: chop_compile is on', r1.device)
    print('r2: chop_script is on', r2.device)
    print('r3: python is on', r3.device)
    print('r4: chop_script2, w/ cuda sync per iter, is on', r4.device)

    assert r1.allclose(r3)
    assert r2.allclose(r3)
    assert r4.allclose(r3)

    t1 = benchmark.Timer(
        stmt = 'chop_torch_compile(X0X1, Ns, results)',
        setup = 'from __main__ import chop_torch_compile',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew, 'results' : results}
    )

    t2 = benchmark.Timer(
        stmt = 'chop_torch_script(X0X1, Ns, results)',
        setup = 'from __main__ import chop_torch_script',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew, 'results' : results}
    )

    t3 = benchmark.Timer(
        stmt = 'chop_torch(X0X1, Ns, results)',
        setup = 'from __main__ import chop_torch',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew, 'results' : results}
    )

    t4 = benchmark.Timer(
        stmt = 'chop_torch_script2(X0X1, Ns, results)',
        setup = 'from __main__ import chop_torch_script2',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew, 'results' : results}
    )

    for i in range(3):
        chop_torch_compile(X0X1new, Nsnew, results)
    print(t1.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch_script(X0X1new, Nsnew, results)
    print(t2.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch(X0X1new, Nsnew, results)
    print(t3.blocked_autorange(min_run_time=1))
    for i in range(3):
        chop_torch_script2(X0X1new, Nsnew, results)
    print(t4.blocked_autorange(min_run_time=1))

if __name__ == '__main__':
    test_device(device='cpu')
    test_device(device='cuda:0')
