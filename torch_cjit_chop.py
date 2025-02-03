import torch
import torch.utils.benchmark as benchmark

from common_dataset import generate_dataset

# torch based
def chop_torch(X0X1, Ns):
    total_size = torch.sum(Ns)

    result = torch.empty((total_size.item(), 6), dtype=torch.float32, device=X0X1.device)

    idx = 0
    for i in range(len(X0X1)):
        size = Ns[i]
        for j in range(3):
            temp = torch.linspace(X0X1[i,j], X0X1[i,j+3], size+1)
            result[idx:idx+size,j] = temp[:-1]
            result[idx:idx+size,j+3] = temp[1:]
        idx += size  # Move the index forward
    return result

batch_chop_X0X1_src = '''\
torch::Tensor batch_chop_X0X1(const torch::Tensor& X0X1, const torch::Tensor& Ns)
{
    torch::AutoGradMode enable_grad(false);
    auto N = Ns.sum(-1).item<int64_t>();
    auto result = torch::empty({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(Ns.device()).requires_grad(false));
    auto length = Ns.size(0);
    for (int64_t i=0, idx=0; i<length; ++i) {
        int64_t n = Ns.index({i}).item<int32_t>();
        for (int64_t j=0; j<3; ++j) {
            auto temp = torch::linspace(X0X1.index({i,j}), X0X1.index({i,j+3}), n+1);
            result.index_put_({torch::indexing::Slice(idx,idx+n),j}, temp.index({torch::indexing::Slice(0,-1)}));
            result.index_put_({torch::indexing::Slice(idx,idx+n),j+3}, temp.index({torch::indexing::Slice(1,torch::indexing::None)}));
        }
        idx += n;
    }
    return result;
}

torch::Tensor batch_chop_X0X1_v2(const torch::Tensor& X0X1, const torch::Tensor& Ns)
{
    torch::AutoGradMode enable_grad(false);
    std::vector<torch::Tensor> tensors;
    auto length = Ns.size(0);
    for (int64_t i=0; i<length; ++i) {
        int64_t n = Ns.index({i}).item<int32_t>();
        auto xs = torch::linspace(X0X1.index({i,0}), X0X1.index({i,3}), n+1, torch::TensorOptions().dtype(torch::kFloat32).device(Ns.device()).requires_grad(false));
        auto ys = torch::linspace(X0X1.index({i,1}), X0X1.index({i,4}), n+1, torch::TensorOptions().dtype(torch::kFloat32).device(Ns.device()).requires_grad(false));
        auto zs = torch::linspace(X0X1.index({i,2}), X0X1.index({i,5}), n+1, torch::TensorOptions().dtype(torch::kFloat32).device(Ns.device()).requires_grad(false));
        tensors.push_back(
            torch::stack({xs.index({torch::indexing::Slice(0,-1)}),
                        ys.index({torch::indexing::Slice(0,-1)}),
                        zs.index({torch::indexing::Slice(0,-1)}),
                        xs.index({torch::indexing::Slice(1,torch::indexing::None)}),
                        ys.index({torch::indexing::Slice(1,torch::indexing::None)}),
                        zs.index({torch::indexing::Slice(1,torch::indexing::None)})},
                        1).view({-1, 6})
                    );
    }
    return torch::cat(tensors, 0);
}
'''

from torch.utils import cpp_extension
cpp_lib = cpp_extension.load_inline(
    name='cpp_lib',
    cpp_sources=batch_chop_X0X1_src,
    extra_cflags=['-O3'],
    extra_include_paths=[
        # `load_inline` needs to know where to find ``pybind11`` headers.
        # os.path.join(os.getenv('CONDA_PREFIX'), 'include')
        '/wcwc/opt/builtin/linux-debian12-x86_64/gcc-12.2.0/python-3.11.9-o4oyrmuslwej6hlaaorfb6zlhefiwpzf/include/python3.11'
    ],
    functions=['batch_chop_X0X1', 'batch_chop_X0X1_v2']
)

# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", cpp_lib.__file__)
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)

module_import_str = f'''\
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", {repr(cpp_lib.__file__)})
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)
'''

X0X1, Ns = generate_dataset()

def test_device(X0X1, Ns, device):
    print('Testing device', device)

    X0X1new = X0X1.to(device)
    Nsnew = Ns.to(device)

    r1 = cpp_lib.batch_chop_X0X1(X0X1new, Nsnew)
    r2 = chop_torch(X0X1new, Nsnew)
    r3 = cpp_lib.batch_chop_X0X1_v2(X0X1new, Nsnew)

    print('r1: cpp_lib is on', r1.device)
    print('r2: python is on', r2.device)
    print('r3: cpp_lib is on', r3.device)

    assert r1.allclose(r2)
    assert r3.allclose(r2)

    t1 = benchmark.Timer(
        stmt = 'cpp_lib.batch_chop_X0X1(X0X1, Ns)',
        setup = module_import_str,
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )
    t3 = benchmark.Timer(
        stmt = 'cpp_lib.batch_chop_X0X1_v2(X0X1, Ns)',
        setup = module_import_str,
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    t2 = benchmark.Timer(
        stmt = 'chop_torch(X0X1, Ns)',
        setup = 'from __main__ import chop_torch',
        globals = {'X0X1' : X0X1new, 'Ns' : Nsnew}
    )

    print(t1.blocked_autorange(min_run_time=1))
    print(t2.blocked_autorange(min_run_time=1))
    print(t3.blocked_autorange(min_run_time=1))

if __name__ == '__main__':
    test_device(X0X1, Ns, device='cpu')
    test_device(X0X1, Ns, device='cuda:0')
