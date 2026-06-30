#!/usr/bin/env python3
"""Generate the candidate task pool for the TileLang benchmark.

Emits task dirs under tasks/<track>/<id>/ with task.json + problem.py (+private
oracle) + solution.py (agent-visible starter). A later validator (validate_tasks.py)
prunes to the verified 100 and writes tasks/manifest.json.

Tracks generated here:
  perf       : optimize a detuned kernel toward a tuned target (score_perf.py)
  implement  : write a kernel from a torch spec, stub start (score_correct.py)
(regression tasks are built separately by gen_regression.py.)
"""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
# Private answer-keys root, OUTSIDE the repo so a browsing agent can't discover it
# via `ls ../..`. Regenerable from this script (the source of truth).
GRADERS = Path(os.environ.get("GRADERS_DIR", Path.home() / ".tl_graders"))

IMPORTS = "import torch\nimport tilelang\nimport tilelang.language as T\n"

# ----------------------------------------------------------------------------
# Kernel-definition fragments. Each returns python source defining `_build`
# (a compiled-kernel factory) for the given dtype. Verified in kernels_probe.py.
# ----------------------------------------------------------------------------

def k_binary(expr_tmpl, dt):
    return f'''DT = "{dt}"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), B: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                a = A[by*bM+i, bx*bN+j]; b = B[by*bM+i, bx*bN+j]
                C[by*bM+i, bx*bN+j] = {expr_tmpl}
    return tilelang.compile(main, out_idx=[2])
'''

def k_unary(expr_tmpl, dt):
    return f'''DT = "{dt}"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                v = A[by*bM+i, bx*bN+j]
                C[by*bM+i, bx*bN+j] = {expr_tmpl}
    return tilelang.compile(main, out_idx=[1])
'''

def k_bias(dt):
    return f'''DT = "{dt}"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), bias: T.Tensor((N,), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[by*bM+i, bx*bN+j] = A[by*bM+i, bx*bN+j] + bias[bx*bN+j]
    return tilelang.compile(main, out_idx=[2])
'''

def k_transpose(dt):
    return f'''DT = "{dt}"
def _build(M, N, bM=32, bN=32, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((N, M), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[bx*bN+j, by*bM+i] = A[by*bM+i, bx*bN+j]
    return tilelang.compile(main, out_idx=[1])
'''

def k_cast():
    return '''DT = "float16"
def _build(M, N, bM=64, bN=64, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                C[by*bM+i, bx*bN+j] = T.Cast("float16", A[by*bM+i, bx*bN+j])
    return tilelang.compile(main, out_idx=[1])
'''

def k_gemv(dt):
    return f'''DT = "{dt}"
def _build(N, K, bN=64):
    @T.prim_func
    def main(A: T.Tensor((K,), DT), B: T.Tensor((N, K), DT), C: T.Tensor((N,), DT)):
        with T.Kernel(T.ceildiv(N, bN), threads=bN) as bn:
            tn = T.get_thread_binding(0)
            acc = T.alloc_local((1,), "float")
            T.clear(acc)
            for k in T.serial(K):
                acc[0] += A[k].astype("float") * B[bn*bN+tn, k].astype("float")
            C[bn*bN+tn] = acc[0].astype(DT)
    return tilelang.compile(main, out_idx=[2])
'''

def k_softmax(dt):
    return f'''DT = "{dt}"
def _build(M, N, bM=1, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            row = T.alloc_fragment((bM, N), "float")
            rmax = T.alloc_fragment((bM,), "float")
            rsum = T.alloc_fragment((bM,), "float")
            T.copy(A[bx*bM, 0], row)
            T.reduce_max(row, rmax, dim=1, clear=True)
            for i, j in T.Parallel(bM, N):
                row[i, j] = T.exp(row[i, j] - rmax[i])
            T.reduce_sum(row, rsum, dim=1)
            for i, j in T.Parallel(bM, N):
                C[bx*bM+i, j] = (row[i, j] / rsum[i]).astype(DT)
    return tilelang.compile(main, out_idx=[1])
'''

def k_gemm(dt):
    acc = "float"
    return f'''DT = "{dt}"
def _build(M, N, K, bM=128, bN=128, bK=64, stages=3, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, K), DT), B: T.Tensor((K, N), DT), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            As = T.alloc_shared((bM, bK), DT); Bs = T.alloc_shared((bK, bN), DT)
            Cl = T.alloc_fragment((bM, bN), "{acc}"); T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by*bM, ko*bK], As); T.copy(B[ko*bK, bx*bN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by*bM, bx*bN])
    return tilelang.compile(main, out_idx=[2])
'''

# data_def: make_inputs + ref (torch ground truth), per family signature.
def data_2d(dt, ref_expr, scale=1.0):
    return f'''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g) * {scale}
    b = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g) * {scale}
    return a, b
def ref(a, b):
    return {ref_expr}
'''

def data_2d_unary(dt, ref_expr, scale=1.0):
    return f'''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g) * {scale}
    return (a,)
def ref(a):
    return {ref_expr}
'''

def data_bias(dt):
    return f'''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g)
    b = torch.randn(N, device="cuda", dtype=torch.{dt}, generator=g)
    return a, b
def ref(a, b):
    return a + b
'''

def data_transpose(dt):
    return f'''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g)
    return (a,)
def ref(a):
    return a.t().contiguous()
'''

def data_cast():
    return '''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.float32, generator=g)
    return (a,)
def ref(a):
    return a.half()
'''

def data_gemv(dt):
    return f'''def make_inputs(N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(K, device="cuda", dtype=torch.{dt}, generator=g)
    b = torch.randn(N, K, device="cuda", dtype=torch.{dt}, generator=g)
    return a, b
def ref(a, b):
    return (b.float() @ a.float()).to(torch.{dt})
'''

def data_softmax(dt):
    return f'''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g)
    return (a,)
def ref(a):
    return torch.softmax(a.float(), dim=1).to(torch.{dt})
'''

def data_gemm(dt):
    return f'''def make_inputs(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.{dt}, generator=g)
    b = torch.randn(K, N, device="cuda", dtype=torch.{dt}, generator=g)
    return a, b
def ref(a, b):
    return a @ b
'''


def write_task(track, tid, prompt, kernel_def, data_def, shapes, *,
               anchors=None, starter=None, tol=(1e-2, 1e-2), measure=None,
               require_tilelang=True):
    d = TASKS / track / tid
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    # problem.py is the ANSWER KEY (kernel oracle + data + anchors/SHAPES). Write it to
    # the PRIVATE graders dir outside the repo, not into the agent-visible task dir.
    prob = IMPORTS + "\n" + kernel_def + "\n" + data_def + f"\nSHAPES = {shapes}\n"
    if anchors:
        prob += anchors
    gdir = GRADERS / tid
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "problem.py").write_text(prob)
    # solution.py: agent-visible starter
    (d / "solution.py").write_text(starter)
    task = {"id": tid, "track": track, "entrypoint": "solution.py",
            "grader": "problem.py", "prompt": prompt,
            "tolerance": {"rtol": tol[0], "atol": tol[1]}}
    if track == "perf":
        task["measure"] = measure or {"warmup": 25, "iters": 50, "rounds": 10,
                                      "rel_std_max": 4.0, "remeasure": 2}
    if track == "implement":
        task["require_tilelang"] = require_tilelang
    (d / "task.json").write_text(json.dumps(task, indent=2))
    return tid


# ---- family registry: (name, kernel_def, data_def, dims, build_call, ref_desc) ----
F16, F32, BF16 = "float16", "float32", "bfloat16"

UNARY = {
    "relu":    ("T.max(v, T.Cast(DT, 0))",                         "torch.relu(a)", 1.0),
    "exp":     ("T.exp(v)",                                        "torch.exp(a)", 0.3),
    "sigmoid": ("T.Cast(DT, 1.0/(1.0+T.exp(-v.astype('float'))))", "torch.sigmoid(a)", 1.0),
    "tanh":    ("T.tanh(v)",                                        "torch.tanh(a)", 1.0),
    "silu":    ("T.Cast(DT, v.astype('float')/(1.0+T.exp(-v.astype('float'))))", "torch.nn.functional.silu(a)", 1.0),
    "square":  ("v * v",                                           "a * a", 1.0),
    "abs":     ("T.abs(v)",                                        "torch.abs(a)", 1.0),
    "neg":     ("-v",                                              "-a", 1.0),
    "leakyrelu": ("T.max(v, v * T.Cast(DT, 0.1))",                 "torch.nn.functional.leaky_relu(a, 0.1)", 1.0),
    "clamp01": ("T.max(T.min(v, T.Cast(DT, 1)), T.Cast(DT, 0))",   "a.clamp(0, 1)", 1.0),
    "relu6":   ("T.min(T.max(v, T.Cast(DT, 0)), T.Cast(DT, 6))",   "a.clamp(0, 6)", 3.0),
    "gelu":    ("T.Cast(DT, 0.5*v.astype('float')*(1.0+T.tanh(0.7978845608*(v.astype('float')+0.044715*v.astype('float')*v.astype('float')*v.astype('float')))))",
                "torch.nn.functional.gelu(a, approximate='tanh')", 1.0),
}
BINARY = {
    "add": ("a + b", "a + b"),
    "mul": ("a * b", "a * b"),
    "sub": ("a - b", "a - b"),
    "max": ("T.max(a, b)", "torch.maximum(a, b)"),
    "min": ("T.min(a, b)", "torch.minimum(a, b)"),
}
# fused binary->unary (reuses the binary kernel template; expr over a,b)
FUSED = {
    "add_relu": ("T.max(a + b, T.Cast(DT, 0))", "torch.relu(a + b)"),
    "mul_relu": ("T.max(a * b, T.Cast(DT, 0))", "torch.relu(a * b)"),
    "abs_diff": ("T.abs(a - b)",                "torch.abs(a - b)"),
    "add_sq":   ("(a + b) * (a + b)",           "(a + b) * (a + b)"),
    "sub_relu": ("T.max(a - b, T.Cast(DT, 0))", "torch.relu(a - b)"),
    "max_relu": ("T.max(T.max(a, b), T.Cast(DT, 0))", "torch.relu(torch.maximum(a, b))"),
    "diff_sq":  ("(a - b) * (a - b)",           "(a - b) * (a - b)"),
    "avg":      ("(a + b) * T.Cast(DT, 0.5)",   "(a + b) * 0.5"),
}


def k_reduce(redop, post, dt):
    """Row reduction MxN -> M.  redop in {reduce_sum, reduce_max}; post applied to r[i]."""
    return f'''DT = "{dt}"
def _build(M, N, bM=1, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M,), DT)):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            row = T.alloc_fragment((bM, N), "float")
            r = T.alloc_fragment((bM,), "float")
            T.copy(A[bx*bM, 0], row)
            T.{redop}(row, r, dim=1, clear=True)
            for i in T.Parallel(bM):
                C[bx*bM+i] = ({post}).astype(DT)
    return tilelang.compile(main, out_idx=[1])
'''

def data_reduce(dt, ref_expr):
    return f'''def make_inputs(M, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, N, device="cuda", dtype=torch.{dt}, generator=g)
    return (a,)
def ref(a):
    return {ref_expr}
'''

REDUCE = {
    "rowsum":  ("reduce_sum", "r[i]",       "a.float().sum(dim=1).to(a.dtype)"),
    "rowmax":  ("reduce_max", "r[i]",       "a.float().amax(dim=1).to(a.dtype)"),
    "rowmean": ("reduce_sum", "r[i] / N",   "a.float().mean(dim=1).to(a.dtype)"),
}


def k_rmsnorm(dt):
    return f'''DT = "{dt}"
def _build(M, N, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
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
'''


def k_layernorm(dt):
    return f'''DT = "{dt}"
def _build(M, N, threads=128):
    @T.prim_func
    def main(A: T.Tensor((M, N), DT), C: T.Tensor((M, N), DT)):
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
'''

NORM = {
    "rmsnorm":   (k_rmsnorm,   "a * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(a.dtype)"),
    "layernorm": (k_layernorm, "torch.nn.functional.layer_norm(a.float(), (a.shape[1],)).to(a.dtype)"),
}

# implement-track case grids (multiple shapes for partial credit; include ragged)
CASES_2D = [(256, 256), (512, 384), (130, 257), (1024, 1024), (64, 4096)]
CASES_GEMV = [(256, 256), (512, 1024), (130, 257), (1024, 512)]
# keep N modest: the reduction oracle holds a (bM, N) row in a fragment, which
# overruns registers for large N. (Optimizing for large N is a harder variant.)
CASES_SOFTMAX = [(64, 256), (128, 512), (32, 512), (96, 128), (256, 384)]
CASES_GEMM_C = [(256, 256, 256), (384, 512, 128), (130, 257, 64)]


def gen_implement():
    ids = []
    for dt in (F16, F32):
        tag = "fp16" if dt == F16 else "fp32"
        for name, (expr, ref_expr, scale) in UNARY.items():
            ids.append(write_task("implement", f"impl_{name}_{tag}",
                f"Implement build(M, N) returning a compiled TileLang kernel that computes "
                f"C = {name}(A) elementwise for an MxN {tag} tensor. kernel(A) -> C.",
                k_unary(expr, dt), data_2d_unary(dt, ref_expr, scale), CASES_2D,
                starter=_stub("build(M, N)", f"C = {name}(A), elementwise, {tag}")))
        for name, (expr, ref_expr) in BINARY.items():
            ids.append(write_task("implement", f"impl_{name}_{tag}",
                f"Implement build(M, N) returning a compiled TileLang kernel computing "
                f"C = A {name} B elementwise for MxN {tag} tensors. kernel(A, B) -> C.",
                k_binary(expr, dt), data_2d(dt, ref_expr), CASES_2D,
                starter=_stub("build(M, N)", f"C = A {name} B, elementwise, {tag}")))
        for name, (expr, ref_expr) in FUSED.items():
            ids.append(write_task("implement", f"impl_{name}_{tag}",
                f"Implement build(M, N) computing C = {name}(A, B) elementwise ({tag}). kernel(A, B) -> C.",
                k_binary(expr, dt), data_2d(dt, ref_expr), CASES_2D,
                starter=_stub("build(M, N)", f"fused {name}(A,B), elementwise, {tag}")))
        for name, (redop, post, ref_expr) in REDUCE.items():
            ids.append(write_task("implement", f"impl_{name}_{tag}",
                f"Implement build(M, N) computing the row {name} of an MxN {tag} tensor "
                f"(output shape [M]). kernel(A) -> C.",
                k_reduce(redop, post, dt), data_reduce(dt, ref_expr), CASES_SOFTMAX,
                tol=(2e-2, 2e-2),
                starter=_stub("build(M, N)", f"row {name} MxN -> [M], {tag}")))
        for name, (kfn, ref_expr) in NORM.items():
            ids.append(write_task("implement", f"impl_{name}_{tag}",
                f"Implement build(M, N) computing row-wise {name} over N of an MxN {tag} "
                f"tensor (output MxN). kernel(A) -> C.",
                kfn(dt), data_2d_unary(dt, ref_expr), CASES_SOFTMAX, tol=(2e-2, 2e-2),
                starter=_stub("build(M, N)", f"row-wise {name} over N, MxN -> MxN, {tag}")))
        ids.append(write_task("implement", f"impl_bias_{tag}",
            f"Implement build(M, N): C[m,n] = A[m,n] + bias[n] ({tag}). kernel(A, bias) -> C.",
            k_bias(dt), data_bias(dt), CASES_2D,
            starter=_stub("build(M, N)", f"row-broadcast bias add, {tag}")))
        ids.append(write_task("implement", f"impl_transpose_{tag}",
            f"Implement build(M, N) computing C = A.T (out shape NxM, {tag}). kernel(A) -> C.",
            k_transpose(dt), data_transpose(dt), CASES_2D,
            starter=_stub("build(M, N)", f"transpose MxN -> NxM, {tag}")))
        ids.append(write_task("implement", f"impl_gemv_{tag}",
            f"Implement build(N, K): C = B @ A where A:[K], B:[N,K], C:[N] ({tag}). kernel(A, B) -> C.",
            k_gemv(dt), data_gemv(dt), CASES_GEMV, tol=(2e-2, 2e-2),
            starter=_stub("build(N, K)", f"gemv C[N]=B[N,K]@A[K], {tag}")))
        ids.append(write_task("implement", f"impl_softmax_{tag}",
            f"Implement build(M, N) computing row softmax over N ({tag}). kernel(A) -> C.",
            k_softmax(dt), data_softmax(dt), CASES_SOFTMAX,
            starter=_stub("build(M, N)", f"row softmax over N, {tag}")))
        ids.append(write_task("implement", f"impl_gemm_{tag}",
            f"Implement build(M, N, K): C = A @ B, A:[M,K] B:[K,N] ({tag}). kernel(A, B) -> C.",
            k_gemm(dt), data_gemm(dt), CASES_GEMM_C, tol=(2e-2, 2e-2),
            starter=_stub("build(M, N, K)", f"matmul C=A@B, {tag}")))
    ids.append(write_task("implement", "impl_cast_f32f16",
        "Implement build(M, N) casting an MxN float32 tensor to float16. kernel(A) -> C.",
        k_cast(), data_cast(), CASES_2D,
        starter=_stub("build(M, N)", "cast float32 -> float16")))
    return ids


def _stub(sig, what):
    return f'''"""Starter solution -- IMPLEMENT THIS. Edit only this file.

Write a TileLang kernel: {what}.
`build(...)` must return a compiled TileLang kernel (tilelang.compile(...)).
Use @T.prim_func + T.Kernel + tilelang.compile(main, out_idx=[...]).
"""
import tilelang
import tilelang.language as T


def {sig}:
    raise NotImplementedError("write your TileLang kernel here")
'''


def gen_perf():
    ids = []
    # GEMM perf: fp16 + bf16 over a shape grid; detuned (small tiles) -> tuned.
    gemm_shapes = {
        "sq2k":  [(2048, 2048, 2048)],
        "sq4k":  [(4096, 4096, 4096)],
        "sq6k":  [(6144, 6144, 6144)],
        "tall":  [(8192, 2048, 2048)],
        "fat":   [(2048, 8192, 2048)],
        "kheavy":[(2048, 2048, 8192)],
        "mix":   [(4096, 4096, 4096), (8192, 8192, 8192)],
        "rect":  [(4096, 2048, 6144)],
        "sq3k":  [(3072, 3072, 3072)],
        "sq5k":  [(5120, 5120, 5120)],
        "wide":  [(2048, 12288, 2048)],
        "deep":  [(2048, 2048, 12288)],
        "sq7k":  [(7168, 7168, 7168)],
        "tall2": [(12288, 2048, 4096)],
        "fat2":  [(2048, 6144, 6144)],
        "kmid":  [(4096, 4096, 2048)],
        "small": [(1024, 1024, 1024)],
        "mix2":  [(3072, 3072, 3072), (6144, 6144, 6144)],
    }
    det = "def baseline(M, N, K):\n    return _build(M, N, K, bM=64, bN=64, bK=32, stages=2, threads=128)\n"
    tun = "def target(M, N, K):\n    return _build(M, N, K, bM=128, bN=128, bK=64, stages=3, threads=128)\n"
    starter_gemm = '''"""Starter -- optimize this slow GEMM. Edit only this file.
build(M, N, K) -> compiled kernel for C=A@B (float16/bf16). kernel(A,B)->C.
Tune bM,bN,bK,stages,threads (and the body) for speed; stay correct."""
import tilelang
import tilelang.language as T


def build(M, N, K):
    DT = "{dt}"
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
'''
    for dt in (F16, BF16):
        tag = "fp16" if dt == F16 else "bf16"
        for sname, shapes in gemm_shapes.items():
            tol = (2e-2, 2e-2) if dt == BF16 else (1e-2, 1e-2)
            ids.append(write_task("perf", f"perf_gemm_{tag}_{sname}",
                f"Optimize the slow {tag} GEMM in solution.py so build(M,N,K) returns a fast, "
                f"correct compiled kernel (C=A@B). Tune tiling/pipeline/threads. Only edit solution.py.",
                k_gemm(dt), data_gemm(dt), shapes,
                anchors=det + tun, starter=starter_gemm.replace("{dt}", dt), tol=tol))
    return ids


def _build_wrapper(sig, args):
    return f"\n\ndef build({sig}):\n    return _build({args})\n"


def write_debug(tid, prompt, right_def, wrong_def, data_def, cases, sig, args, tol):
    """Debug task: solution.py is a compilable-but-WRONG kernel; problem.py holds the
    correct oracle. Scored by score_correct (fraction of cases the fix makes correct)."""
    d = TASKS / "debug" / tid
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    gdir = GRADERS / tid
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "problem.py").write_text(
        IMPORTS + "\n" + right_def + "\n" + data_def + f"\nSHAPES = {cases}\n")
    starter = ('"""This TileLang kernel has a BUG -- fix it. Edit only this file.\n\n'
               f'Spec: {prompt}\nThe kernel compiles but produces wrong results for some '
               'inputs. Find and fix the defect so the kernel output matches the spec.\n"""\n'
               + IMPORTS + "\n" + wrong_def + _build_wrapper(sig, args))
    (d / "solution.py").write_text(starter)
    task = {"id": tid, "track": "debug", "entrypoint": "solution.py", "grader": "problem.py",
            "prompt": "A TileLang kernel in solution.py has a bug. " + prompt
            + " Fix solution.py so the kernel is correct. Only edit solution.py.",
            "tolerance": {"rtol": tol[0], "atol": tol[1]}, "require_tilelang": True}
    (d / "task.json").write_text(json.dumps(task, indent=2))
    return tid


def k_gemm_buggy(dt):
    """gemm missing the accumulator clear -> accumulates uninitialized garbage."""
    return k_gemm(dt).replace('Cl = T.alloc_fragment((bM, bN), "float"); T.clear(Cl)',
                              'Cl = T.alloc_fragment((bM, bN), "float")')


def gen_debug():
    ids = []
    dbg2d = [(256, 256), (512, 384), (130, 257), (1024, 1024)]
    for dt in (F16, F32):
        tag = "fp16" if dt == F16 else "fp32"
        for name, right, wrong, refx in [
            ("add", "a + b", "a - b", "a + b"),
            ("mul", "a * b", "a + b", "a * b"),
            ("sub", "a - b", "b - a", "a - b"),
        ]:
            ids.append(write_debug(f"debug_{name}_{tag}",
                f"It should compute C = A {name} B elementwise ({tag}).",
                k_binary(right, dt), k_binary(wrong, dt), data_2d(dt, refx), dbg2d,
                "M, N", "M, N", (1e-2, 1e-2)))
        for name, right, wrong, refx in [
            ("relu", "T.max(v, T.Cast(DT, 0))", "v", "torch.relu(a)"),
            ("square", "v * v", "v", "a * a"),
            ("abs", "T.abs(v)", "v", "torch.abs(a)"),
        ]:
            ids.append(write_debug(f"debug_{name}_{tag}",
                f"It should compute C = {name}(A) elementwise ({tag}).",
                k_unary(right, dt), k_unary(wrong, dt), data_2d_unary(dt, refx), dbg2d,
                "M, N", "M, N", (1e-2, 1e-2)))
        ids.append(write_debug(f"debug_gemm_noclear_{tag}",
            f"It should compute C = A @ B ({tag}).",
            k_gemm(dt), k_gemm_buggy(dt), data_gemm(dt),
            [(256, 256, 256), (384, 512, 128), (130, 257, 64)],
            "M, N, K", "M, N, K", (2e-2, 2e-2)))
    return ids


def main():
    perf = gen_perf()
    impl = gen_implement()
    dbg = gen_debug()
    manifest = {"perf_candidates": perf, "implement_candidates": impl,
                "debug_candidates": dbg,
                "counts": {"perf": len(perf), "implement": len(impl), "debug": len(dbg)}}
    (TASKS / "candidates.json").write_text(json.dumps(manifest, indent=2))
    print(f"generated perf={len(perf)} implement={len(impl)} debug={len(dbg)} "
          f"total={len(perf)+len(impl)+len(dbg)} candidates")
    print("wrote", TASKS / "candidates.json")


if __name__ == "__main__":
    main()
