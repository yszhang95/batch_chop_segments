import torch
import torch.profiler

@torch.jit.script
def pow2(X):
    for i in range(len(X)):
        X[i] =X[i]**2
    return X
@torch.jit.script
def pow2_5times(X):
    for i in range(5):
        X = X**2
    return X

X = [torch.rand((i,)) for i in range(10, 20, 1)]
for i in range(len(X)):
    pow2(X[i])
    print(pow2.graph_for(X[i]))
print(pow2_5times.graph_for(X[0]))

X = [torch.rand((i,)) for i in range(200, 300,2)]

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
    for i in range(len(X)):
        pow2(X[i])

prof.export_chrome_trace('trace.json')

