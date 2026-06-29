import statistics as st
import torch
import tilelang
import tilelang.language as T


def matmul(M, N, K, bM, bN, bK, dtype="float16", accum="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            As = T.alloc_shared((bM, bK), dtype)
            Bs = T.alloc_shared((bK, bN), dtype)
            Cl = T.alloc_fragment((bM, bN), accum)
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=3):
                T.copy(A[by * bM, ko * bK], As)
                T.copy(B[ko * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * bM, bx * bN])
    return main


def bench(fn, iters=50):
    """Median latency (ms) over `iters` launches, timed with CUDA events."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return st.median(times)


def relstd(xs):
    return 100 * st.pstdev(xs) / st.mean(xs)


def run_shape(M, N, K, rounds=15):
    kernel = tilelang.compile(matmul(M, N, K, 128, 128, 32), out_idx=[2])
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = kernel(a, b)
    ref = a @ b
    err = (c.float() - ref.float()).abs().max().item()

    til = lambda: kernel(a, b)
    tor = lambda: torch.matmul(a, b)
    # warmup to reach steady clock state
    for _ in range(30):
        til(); tor()
    torch.cuda.synchronize()

    til_t, tor_t, ratio = [], [], []
    for _ in range(rounds):
        t = bench(tor)          # interleaved: measure both adjacent in time
        u = bench(til)
        tor_t.append(t); til_t.append(u); ratio.append(t / u)

    flop = 2 * M * N * K
    tflops = flop / (st.median(til_t) * 1e-3) / 1e12
    print(f"\n=== {M}x{N}x{K} (maxerr={err:.3g}) ===")
    print(f"  torch     median={st.median(tor_t):.4f}ms  relstd={relstd(tor_t):.2f}%")
    print(f"  tilelang  median={st.median(til_t):.4f}ms  relstd={relstd(til_t):.2f}%  ({tflops:.0f} TFLOP/s)")
    print(f"  RATIO torch/tilelang  mean={st.mean(ratio):.3f}  relstd={relstd(ratio):.2f}%  "
          f"min={min(ratio):.3f} max={max(ratio):.3f}")


if __name__ == "__main__":
    print("tilelang", tilelang.__version__, "torch", torch.__version__,
          "dev", torch.cuda.get_device_name(0))
    for shape in [(4096, 4096, 4096), (8192, 8192, 8192)]:
        run_shape(*shape)
