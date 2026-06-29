"""This TileLang kernel has a BUG -- fix it. Edit only this file.

Spec: It should compute C = A add B elementwise (fp16).
The kernel compiles but produces wrong results for some inputs. Find and fix the defect so the kernel output matches the spec.
"""
import torch
import tilelang
import tilelang.language as T

DT = "float16"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), B: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                a = A[by*bM+i, bx*bN+j]; b = B[by*bM+i, bx*bN+j]
                C[by*bM+i, bx*bN+j] = a - b
    return tilelang.compile(main, out_idx=[2])


def build(M, N):
    return _build(M, N)
