import torch
import tilelang
import tilelang.language as T

DT = "float32"
def _build(M, N, bM=8, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            row = T.alloc_fragment((bM, N), "float")
            rmax = T.alloc_fragment((bM,), "float")
            rsum = T.alloc_fragment((bM,), "float")
            T.copy(A[bx*bM, 0], row)
            T.reduce_max(row, rmax, dim=1, clear=True)
            for i, j in T.Parallel(bM, N):
                row[i, j] = T.exp(row[i, j] - rmax[i])
            T.reduce_sum(row, rsum, dim=1)
            for i, j in T.Parallel(bM, N):
                C[bx*bM+i, j] = (row[i, j] / rsum[i]).astype(DT)
    return tilelang.compile(main, out_idx=[1])

def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.float32, generator=g)
    return (a,)
def ref(a):
    return torch.softmax(a.float(), dim=1).to(torch.float32)

SHAPES = [(64, 256), (128, 512), (32, 512), (96, 128), (256, 384)]
