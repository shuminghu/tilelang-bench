"""Verify candidate kernel templates compile + are correct on tilelang 0.1.11.

Run on a free GPU:  CUDA_VISIBLE_DEVICES=6 .venv/bin/python harness/kernels_probe.py
Prints PASS/FAIL per template. Only PASSing templates get baked into gen_tasks.py.
"""
import torch
import tilelang
import tilelang.language as T

tol = dict(rtol=1e-2, atol=1e-2)
results = []


def check(name, build, ref_fn, shape, mk):
    try:
        a = mk(*shape)
        k = build(*shape)
        out = k(*a) if isinstance(a, tuple) else k(a)
        torch.cuda.synchronize()
        ref = ref_fn(*a) if isinstance(a, tuple) else ref_fn(a)
        ok = torch.allclose(out.float(), ref.float(), **tol)
        err = (out.float() - ref.float()).abs().max().item()
        results.append((name, ok, f"max_err={err:.3g}"))
    except Exception as e:
        import traceback
        results.append((name, False, f"{type(e).__name__}: {str(e)[:120]}"))


# ---------- pointwise binary: C = A + B ----------
def build_add(M, N, bM=64, bN=64, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), B: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[by * bM + i, bx * bN + j] = A[by * bM + i, bx * bN + j] + B[by * bM + i, bx * bN + j]
    return tilelang.compile(main, out_idx=[2])


# ---------- pointwise unary: C = relu(A) ----------
def build_relu(M, N, bM=64, bN=64, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                v = A[by * bM + i, bx * bN + j]
                C[by * bM + i, bx * bN + j] = T.max(v, T.Cast(dt, 0))
    return tilelang.compile(main, out_idx=[1])


# ---------- broadcast bias add: C[m,n] = A[m,n] + bias[n] ----------
def build_bias(M, N, bM=64, bN=64, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), bias: T.Tensor((N,), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[by * bM + i, bx * bN + j] = A[by * bM + i, bx * bN + j] + bias[bx * bN + j]
    return tilelang.compile(main, out_idx=[2])


# ---------- transpose: C[n,m] = A[m,n] ----------
def build_transpose(M, N, bM=32, bN=32, threads=128, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), C: T.Tensor((N, M), dt)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[bx * bN + j, by * bM + i] = A[by * bM + i, bx * bN + j]
    return tilelang.compile(main, out_idx=[1])


# ---------- cast: fp32 -> fp16 ----------
def build_cast(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[by * bM + i, bx * bN + j] = T.Cast("float16", A[by * bM + i, bx * bN + j])
    return tilelang.compile(main, out_idx=[1])


# ---------- gemv: C[n] = sum_k A[k]*B[n,k]  (B is [N,K]) ----------
def build_gemv(N, K, bN=64, bK=64, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((K,), dt), B: T.Tensor((N, K), dt), C: T.Tensor((N,), dt)):
        with T.Kernel(T.ceildiv(N, bN), threads=bN) as bn:
            tn = T.get_thread_binding(0)
            acc = T.alloc_local((1,), "float")
            T.clear(acc)
            for k in T.serial(K):
                acc[0] += A[k].astype("float") * B[bn * bN + tn, k].astype("float")
            C[bn * bN + tn] = acc[0].astype(dt)
    return tilelang.compile(main, out_idx=[2])


# ---------- row softmax over N ----------
def build_softmax(M, N, bM=8, dt="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dt), C: T.Tensor((M, N), dt)):
        with T.Kernel(T.ceildiv(M, bM), threads=128) as bx:
            row = T.alloc_fragment((bM, N), "float")
            rmax = T.alloc_fragment((bM,), "float")
            rsum = T.alloc_fragment((bM,), "float")
            T.copy(A[bx * bM, 0], row)
            T.reduce_max(row, rmax, dim=1, clear=True)
            for i, j in T.Parallel(bM, N):
                row[i, j] = T.exp(row[i, j] - rmax[i])
            T.reduce_sum(row, rsum, dim=1)
            for i, j in T.Parallel(bM, N):
                C[bx * bM + i, j] = (row[i, j] / rsum[i]).astype(dt)
    return tilelang.compile(main, out_idx=[1])


dev = "cuda"
f16 = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16)
f32 = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32)

check("add", build_add, lambda a, b: a + b, (512, 512), lambda M, N: (f16(M, N), f16(M, N)))
check("relu", build_relu, lambda a: torch.relu(a), (512, 512), lambda M, N: f16(M, N))
check("bias", build_bias, lambda a, b: a + b, (512, 512), lambda M, N: (f16(M, N), f16(N)))
check("transpose", build_transpose, lambda a: a.t().contiguous(), (256, 512), lambda M, N: f16(M, N))
check("cast", build_cast, lambda a: a.half(), (512, 512), lambda M, N: f32(M, N))
check("gemv", build_gemv, lambda a, b: (b.float() @ a.float()), (512, 1024), lambda N, K: (f16(K), f16(N, K)))
check("softmax", build_softmax, lambda a: torch.softmax(a.float(), dim=1), (64, 512), lambda M, N: f16(M, N))

print("\n=== kernel probe results ===")
for name, ok, msg in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:12s} {msg}")
print("PASS:", ",".join(n for n, ok, _ in results if ok))
