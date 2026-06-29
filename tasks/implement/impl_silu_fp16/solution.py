"""Starter solution -- IMPLEMENT THIS. Edit only this file.

Write a TileLang kernel: C = silu(A), elementwise, fp16.
`build(...)` must return a compiled TileLang kernel (tilelang.compile(...)).
Use @T.prim_func + T.Kernel + tilelang.compile(main, out_idx=[...]).
"""
import tilelang
import tilelang.language as T


def build(M, N):
    raise NotImplementedError("write your TileLang kernel here")
