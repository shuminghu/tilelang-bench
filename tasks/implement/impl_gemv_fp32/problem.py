import torch
import tilelang
import tilelang.language as T

DT = "float32"
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

def make_inputs(N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(K, device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn(N, K, device="cuda", dtype=torch.float32, generator=g)
    return a, b
def ref(a, b):
    return (b.float() @ a.float()).to(torch.float32)

SHAPES = [(256, 256), (512, 1024), (130, 257), (1024, 512)]
