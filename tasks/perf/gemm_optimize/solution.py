"""Starter solution for gemm_optimize. THIS IS THE FILE YOU EDIT.

You are given a working but slow fp16 GEMM kernel written in TileLang.
Your job: make `build(M, N, K)` return a compiled kernel that computes
C = A @ B (A: [M,K], B: [K,N], both float16) as fast as possible while
staying numerically correct.

The returned object must be callable as `kernel(A, B) -> C`.

Things worth tuning: block sizes (bM, bN, bK), pipeline stages, thread count,
L2 swizzling (T.use_swizzle), layout annotations, etc.
"""
import tilelang
import tilelang.language as T


def build(M, N, K):
    # TODO: optimize these parameters / the kernel body.
    bM, bN, bK, stages, threads = 64, 64, 32, 2, 128

    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bK, bN), "float16")
            Cl = T.alloc_fragment((bM, bN), "float")
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by * bM, ko * bK], As)
                T.copy(B[ko * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * bM, bx * bN])

    return tilelang.compile(main, out_idx=[2])
