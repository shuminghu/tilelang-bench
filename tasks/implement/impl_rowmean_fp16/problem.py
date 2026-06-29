import torch
import tilelang
import tilelang.language as T

DT = "float16"
def _build(M, N, bM=8, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M,), DT)):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            row = T.alloc_fragment((bM, N), "float")
            r = T.alloc_fragment((bM,), "float")
            T.copy(A[bx*bM, 0], row)
            T.reduce_sum(row, r, dim=1, clear=True)
            for i in T.Parallel(bM):
                C[bx*bM+i] = (r[i] / N).astype(DT)
    return tilelang.compile(main, out_idx=[1])

def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.float16, generator=g)
    return (a,)
def ref(a):
    return a.float().mean(dim=1).to(a.dtype)

SHAPES = [(64, 256), (128, 512), (32, 512), (96, 128), (256, 384)]
