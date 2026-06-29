import statistics as st
import torch
import tilelang
import tilelang.language as T


def matmul(M, N, K, bM, bN, bK, stages, dtype="float16", accum="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            As = T.alloc_shared((bM, bK), dtype)
            Bs = T.alloc_shared((bK, bN), dtype)
            Cl = T.alloc_fragment((bM, bN), accum)
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by * bM, ko * bK], As)
                T.copy(B[ko * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * bM, bx * bN])
    return main


def bench(fn, iters=50):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    return st.median(t)


def relstd(xs):
    return 100 * st.pstdev(xs) / st.mean(xs)


M = N = K = 4096
# "slow" baseline config vs "fast" target config -- both tilelang
slow = tilelang.compile(matmul(M, N, K, 64, 64, 32, 2), out_idx=[2])
fast = tilelang.compile(matmul(M, N, K, 128, 128, 64, 3), out_idx=[2])
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
fs, sl = lambda: fast(a, b), lambda: slow(a, b)
for _ in range(30):
    fs(); sl()
torch.cuda.synchronize()

ratio = []   # speedup of fast over slow == what a perf score normalizes
for _ in range(20):
    s = bench(sl); f = bench(fs); ratio.append(s / f)
print("tilelang-vs-tilelang  speedup(slow/fast)  "
      f"mean={st.mean(ratio):.3f}  relstd={relstd(ratio):.2f}%  "
      f"min={min(ratio):.3f}  max={max(ratio):.3f}")
