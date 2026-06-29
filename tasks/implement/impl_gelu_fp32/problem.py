import torch
import tilelang
import tilelang.language as T

DT = "float32"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                v = A[by*bM+i, bx*bN+j]
                C[by*bM+i, bx*bN+j] = T.Cast(DT, 0.5*v.astype('float')*(1.0+T.tanh(0.7978845608*(v.astype('float')+0.044715*v.astype('float')*v.astype('float')*v.astype('float')))))
    return tilelang.compile(main, out_idx=[1])

def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.float32, generator=g) * 1.0
    return (a,)
def ref(a):
    return torch.nn.functional.gelu(a, approximate='tanh')

SHAPES = [(256, 256), (512, 384), (130, 257), (1024, 1024), (64, 4096)]
