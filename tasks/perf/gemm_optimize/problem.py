"""Harness-owned grader fixture for the gemm_optimize task.

Defines the problem (shapes, inputs, reference output) and the two scoring
anchors: `baseline` (naive kernel, score 0.0) and `target` (tuned tilelang
kernel, score 1.0, per normalization option (a)). NOT shown to the agent.
"""
import torch
import tilelang
import tilelang.language as T

SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192)]


def make_inputs(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16, generator=g)
    return a, b


def ref(a, b):
    return a @ b


def _mm(M, N, K, bM, bN, bK, stages, threads=128):
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
    return main


def baseline(M, N, K):
    """Naive small-tile kernel -> score 0.0 anchor (== the starter solution)."""
    return tilelang.compile(_mm(M, N, K, 64, 64, 32, 2), out_idx=[2])


def target(M, N, K):
    """Tuned tilelang kernel -> score 1.0 anchor."""
    return tilelang.compile(_mm(M, N, K, 128, 128, 64, 3), out_idx=[2])
