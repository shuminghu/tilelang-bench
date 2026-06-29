import torch
import tilelang
import tilelang.language as T

DT = "float16"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                v = A[by*bM+i, bx*bN+j]
                C[by*bM+i, bx*bN+j] = T.max(v, v * T.Cast(DT, 0.1))
    return tilelang.compile(main, out_idx=[1])

def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.float16, generator=g) * 1.0
    return (a,)
def ref(a):
    return torch.nn.functional.leaky_relu(a, 0.1)

SHAPES = [(256, 256), (512, 384), (130, 257), (1024, 1024), (64, 4096)]
