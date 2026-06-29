"""This TileLang kernel has a BUG -- fix it. Edit only this file.

Spec: It should compute C = A @ B (fp32).
The kernel compiles but produces wrong results for some inputs. Find and fix the defect so the kernel output matches the spec.
"""
import torch
import tilelang
import tilelang.language as T

DT = "float32"
def _build(M, N, K, bM=128, bN=128, bK=64, stages=3, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            As = T.alloc_shared((bM, bK), DT); Bs = T.alloc_shared((bK, bN), DT)
            Cl = T.alloc_fragment((bM, bN), "float")
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by*bM, ko*bK], As); T.copy(B[ko*bK, bx*bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by*bM, bx*bN])
    return tilelang.compile(main, out_idx=[2])


def build(M, N, K):
    return _build(M, N, K)
