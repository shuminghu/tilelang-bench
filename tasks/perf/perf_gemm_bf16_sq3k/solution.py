"""Starter -- optimize this slow GEMM. Edit only this file.
build(M, N, K) -> compiled kernel for C=A@B (float16/bf16). kernel(A,B)->C.
Tune bM,bN,bK,stages,threads (and the body) for speed; stay correct."""
import tilelang
import tilelang.language as T


def build(M, N, K):
    DT = "bfloat16"
    bM, bN, bK, stages, threads = 64, 64, 32, 2, 128
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
