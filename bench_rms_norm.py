import torch
import torch.nn as nn
import timeit
import time

B, S, H = 32, 2048, 1024

timeit_iters = 20
repeat_count = 4

norm = nn.RMSNorm(H, elementwise_affine=True).to("cuda")
norm_no_weight = nn.RMSNorm(H, elementwise_affine=False).to("cuda")

hidden = torch.randn((B, S, H), device="cuda")
weight = torch.randn((1,1,H), device="cuda")
scale = torch.randn((B,S,1), device="cuda")

def benchmark(fn, name):
    def sync(fn):
        fn()
        torch.cuda.synchronize()

    torch.cuda.synchronize()

    execution_time = min(timeit.repeat(lambda: sync(fn), repeat=repeat_count, number=timeit_iters))
    print(f"{name}: {1e6 * execution_time / timeit_iters:.8f}us")

    torch.cuda.empty_cache()
    time.sleep(0.1)
    torch.cuda.synchronize()

@torch.compile()
def fn_add(x):
    return x + 1

@torch.compile()
def fn_mul(x, y):
    return x * y

fn_add(hidden)
fn_mul(hidden, weight)

norm_compiled = torch.compile(norm)
norm_compiled(hidden)

norm_no_weight_compiled = torch.compile(norm_no_weight)
norm_no_weight_compiled(hidden)


benchmark(lambda: norm(hidden), "RMSNorm")
benchmark(lambda: norm_compiled(hidden), "RMSNorm C")
benchmark(lambda: norm_no_weight(hidden), "RMSNorm no weight")
benchmark(lambda: norm_no_weight_compiled(hidden), "RMSNorm no weight C")
benchmark(lambda: hidden * weight, "Elementwise Multiplication")
benchmark(lambda: fn_mul, "Elementwise Multiplication C")
benchmark(lambda: hidden + 1, "Elementwise Additionm")
benchmark(lambda: fn_add, "Elementwise Addition C")
# benchmark(lambda: hidden * scale, "Scaling first 2 dim")
