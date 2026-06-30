"""Verify the FIXED reduction/norm kernels (bM=1 row-per-block, per the repo's
rms_norm pattern) compile + are correct across the planned case shapes.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python harness/kernels_probe2.py
"""
import torch
import tilelang
import tilelang.language as T

tol = dict(rtol=2e-2, atol=2e-2)
results = []
CASES = [(64, 256), (128, 512), (32, 512), (96, 128), (256, 384), (256, 2048)]


def check(name, build, ref_fn, dt):
    for (M, N) in CASES:
        try:
            a = torch.randn(M, N, device="cuda", dtype=dt)
            k = build(M, N)
            out = k(a)
            torch.cuda.synchronize()
            ref = ref_fn(a)
            ok = torch.allclose(out.float(), ref.float(), **tol)
            if not ok:
                err = (out.float() - ref.float()).abs().max().item()
                results.append((f"{name}[{M}x{N}]", False, f"max_err={err:.3g}")); continue
        except Exception as e:
            results.append((f"{name}[{M}x{N}]", False, f"{type(e).__name__}: {str(e)[:90]}")); continue
    results.append((name, True, "all cases ok"))


def softmax(M, N, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(M, threads=threads) as bx:
            row = T.alloc_fragment((1, N), "float")
            rmax = T.alloc_fragment((1,), "float")
            rsum = T.alloc_fragment((1,), "float")
            T.copy(A[bx, 0], row)
            T.reduce_max(row, rmax, dim=1, clear=True)
            for i, j in T.Parallel(1, N):
                row[i, j] = T.exp(row[i, j] - rmax[i])
            T.reduce_sum(row, rsum, dim=1)
            for i, j in T.Parallel(1, N):
                row[i, j] = row[i, j] / rsum[i]
            T.copy(row, C[bx, 0])
    return tilelang.compile(main, out_idx=[1])


def reduce_row(redop, post):
    def build(M, N, threads=128, dt="float16"):
        @T.prim_func
        def main(A: T.Tensor((M, N), dt), C: T.Tensor((M,), dt)):
            with T.Kernel(M, threads=threads) as bx:
                row = T.alloc_fragment((1, N), "float")
                r = T.alloc_fragment((1,), "float")
                T.copy(A[bx, 0], row)
                getattr(T, redop)(row, r, dim=1, clear=True)
                C[bx] = T.Cast(dt, eval(post))
        return tilelang.compile(main, out_idx=[1])
    return build


def rmsnorm(M, N, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(M, threads=threads) as bx:
            x = T.alloc_fragment((1, N), "float")
            xs = T.alloc_fragment((1, N), "float")
            s = T.alloc_fragment((1,), "float")
            T.copy(A[bx, 0], x)
            for i, j in T.Parallel(1, N):
                xs[i, j] = x[i, j] * x[i, j]
            T.reduce_sum(xs, s, dim=1)
            for i in T.Parallel(1):
                s[i] = T.rsqrt(s[i] / N + 1e-6)
            for i, j in T.Parallel(1, N):
                x[i, j] = x[i, j] * s[i]
            T.copy(x, C[bx, 0])
    return tilelang.compile(main, out_idx=[1])


def layernorm(M, N, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(M, threads=threads) as bx:
            x = T.alloc_fragment((1, N), "float")
            c = T.alloc_fragment((1, N), "float")
            mean = T.alloc_fragment((1,), "float")
            var = T.alloc_fragment((1,), "float")
            T.copy(A[bx, 0], x)
            T.reduce_sum(x, mean, dim=1)
            for i in T.Parallel(1):
                mean[i] = mean[i] / N
            for i, j in T.Parallel(1, N):
                c[i, j] = (x[i, j] - mean[i]) * (x[i, j] - mean[i])
            T.reduce_sum(c, var, dim=1)
            for i in T.Parallel(1):
                var[i] = T.rsqrt(var[i] / N + 1e-5)
            for i, j in T.Parallel(1, N):
                x[i, j] = (x[i, j] - mean[i]) * var[i]
            T.copy(x, C[bx, 0])
    return tilelang.compile(main, out_idx=[1])


for dt, t in [(torch.float16, "float16"), (torch.float32, "float32")]:
    tag = "fp16" if dt == torch.float16 else "fp32"
    check(f"softmax_{tag}", lambda M, N, t=t: softmax(M, N, dt=t),
          lambda a: torch.softmax(a.float(), dim=1), dt)
    check(f"rowsum_{tag}", lambda M, N, t=t: reduce_row("reduce_sum", "r[0]")(M, N, dt=t),
          lambda a: a.float().sum(1), dt)
    check(f"rowmax_{tag}", lambda M, N, t=t: reduce_row("reduce_max", "r[0]")(M, N, dt=t),
          lambda a: a.float().amax(1), dt)
    check(f"rowmean_{tag}", lambda M, N, t=t: reduce_row("reduce_sum", "r[0] / N")(M, N, dt=t),
          lambda a: a.float().mean(1), dt)
    check(f"rmsnorm_{tag}", lambda M, N, t=t: rmsnorm(M, N, dt=t),
          lambda a: a * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(a.dtype), dt)
    check(f"layernorm_{tag}", lambda M, N, t=t: layernorm(M, N, dt=t),
          lambda a: torch.nn.functional.layer_norm(a.float(), (a.shape[1],)), dt)

print("\n=== reduction/norm probe ===")
for name, ok, msg in results:
    if not name.endswith("]") or not ok:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:16s} {msg}")
print("PASS families:", ",".join(n for n, ok, m in results if ok and m == "all cases ok"))
