import torch
import tilelang
import tilelang.language as T

DT = "float16"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), bias: T.Tensor((N,), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[by*bM+i, bx*bN+j] = A[by*bM+i, bx*bN+j] + bias[bx*bN+j]
    return tilelang.compile(main, out_idx=[2])

def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn(N, device="cuda", dtype=torch.float16, generator=g)
    return a, b
def ref(a, b):
    return a + b

SHAPES = [(256, 256), (512, 384), (130, 257), (1024, 1024), (64, 4096)]
