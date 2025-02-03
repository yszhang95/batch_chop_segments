import torch

def __init__():
    pass

def generate_dataset(device='cpu'):
    torch.manual_seed(1024)
    X0 = torch.rand((10_000, 3), requires_grad=False, device=device)
    X1 = (torch.rand((10_000, 3), requires_grad=False, device=device) + 0.5) * 2
    step_limit = 1
    X0X1 = torch.cat([X0,X1], dim=1)
    LdX = torch.linalg.norm(X1-X0, dim=1)
    Ns = (LdX//step_limit + 1).to(torch.int32)
    return X0X1, Ns

