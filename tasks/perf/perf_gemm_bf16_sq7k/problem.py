import torch
import tilelang
import tilelang.language as T

DT = "bfloat16"
def _build(M, N, K, bM=128, bN=128, bK=64, stages=3, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            As = T.alloc_shared((bM, bK), DT); Bs = T.alloc_shared((bK, bN), DT)
            Cl = T.alloc_fragment((bM, bN), "float"); T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by*bM, ko*bK], As); T.copy(B[ko*bK, bx*bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by*bM, bx*bN])
    return tilelang.compile(main, out_idx=[2])

def make_inputs(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16, generator=g)
    return a, b
def ref(a, b):
    return a @ b

SHAPES = [(7168, 7168, 7168)]
def baseline(M, N, K):
    return _build(M, N, K, bM=64, bN=64, bK=32, stages=2, threads=128)
def target(M, N, K):
    return _build(M, N, K, bM=128, bN=128, bK=64, stages=3, threads=128)
