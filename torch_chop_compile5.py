import torch
import torch.utils.benchmark as benchmark
from common_dataset import generate_dataset

# torch based
def chop_seg(seg, size):
    xs = torch.linspace(seg[0], seg[3], size+1, device=seg.device)
    ys = torch.linspace(seg[1], seg[4], size+1, device=seg.device)
    zs = torch.linspace(seg[2], seg[5], size+1, device=seg.device)
    return torch.stack([xs[:-1], ys[:-1], zs[:-1], xs[1:], ys[1:], zs[1:]], dim=1).view(-1, 6)

chop_seg_compile = torch.compile(chop_seg)
chop_seg_script = torch.jit.script(chop_seg)

def chop_torch(X0X1, Ns):
    output = []
    for i in range(len(Ns)):
        size = Ns[i]
        output.append(chop_seg_compile(X0X1[i], size))
    return torch.cat(output, dim=0)

def chop_torch2(X0X1, Ns):
    output = []
    for i in range(len(Ns)):
        size = Ns[i]
        output.append(chop_seg_script(X0X1[i], size))
    return torch.cat(output, dim=0)

def chop_torch3(X0X1, Ns):
    output = []
    for i in range(len(Ns)):
        size = Ns[i]
        output.append(chop_seg(X0X1[i], size))
    return torch.cat(output, dim=0)

chop_torch_compile = torch.compile(chop_torch3)
chop_torch_script = torch.jit.script(chop_torch3)

def test_device(device):
    print('Testing device', device)

    X0X1, Ns = generate_dataset()
    X0X1new = X0X1.to(device)
    Nsnew = Ns.to(device)

    r1 = chop_torch_compile(X0X1new, Nsnew)
    r2 = chop_torch_script(X0X1new, Nsnew)
    r3 = chop_torch(X0X1new, Nsnew)
    r4 = chop_torch2(X0X1new, Nsnew)

    print('r1: chop_compile is on', r1.device)
    print('r2: chop_script is on', r2.device)
    print('r3: python is on', r3.device)
    print('r4: python+script is on', r4.device)

    assert r1.allclose(r3)
    assert r2.allclose(r3)

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
        stmt = 'chop_torch2(X0X1, Ns)',
        setup = 'from __main__ import chop_torch2',
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
        chop_torch2(X0X1new, Nsnew)
    print(t4.blocked_autorange(min_run_time=1))

if __name__ == '__main__':
    test_device(device='cpu')
    test_device(device='cuda:0')
