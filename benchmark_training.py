import itertools
import math
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
from tqdm import tqdm


def attention_pytorch_eager(query, key, value, is_causal=False, attn_mask=None, dropout_p=0.1):
    L = query.size(-1)
    S = value.size(-1)
    attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    if attn_mask is None:
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(L)), dim=-1)
    else:
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(L)) + attn_mask, dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def attention_pytorch_native(query, key, value, is_causal=False, attn_mask=None, dropout_p=0.1):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
    )


def benchmark_forward_backward(func, n_repeat: int, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_repeat):
        res = func(**kwargs)
        loss = res.sum()
        loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event)) / n_repeat


torch.manual_seed(0)
n_repeat = 30
# batch_size = 8
# seqlen = 512
# nheads = 12
# headdim = 16
dropout_p = 0.0
causal = False
dtype = torch.float16  # torch.float32 is not supported for Hazy-Research implementation
device = "cuda"

all_parameters = {
    "batch_size": [8, 16, 64],
    "seq_len": [64, 128, 256, 512, 1024],
    "head_dim": [32, 64, 128],
    "num_heads": [12, 16, 24],
}


def grid(
    parameters: Dict[str, Iterable[Any]],
) -> Iterable:
    for params in itertools.product(*parameters.values()):
        returned_list = list(params)
        yield returned_list


output_file = open("benchmark_training_attention.csv", "w")
output_file.write(
    "batch_size, seq_len, headdim, nheads, PT eager (ms/forward), PT native (ms/forward), Native speedup\n"
)


class BenchmarkModel(nn.Module):
    def __init__(self, batch_size, nheads, seqlen, headdim, device, dtype):
        super().__init__()
        qkv = torch.randn(batch_size, 3, nheads, seqlen, headdim, device=device, dtype=dtype)
        query, key, value = qkv.unbind(dim=1)
        self.query = nn.Parameter(query)
        self.key = nn.Parameter(key)
        self.value = nn.Parameter(value)

    def forward(self, attention_func, dropout_p=0.1):
        return attention_func(query=self.query, key=self.key, value=self.value, dropout_p=dropout_p)


for params in tqdm(list(grid(all_parameters))):
    batch_size, seqlen, headdim, nheads = tuple(params)
    print(f"Running: bs={batch_size}, seqlen={seqlen}, headdim={headdim}, nheads={nheads}")

    model = BenchmarkModel(batch_size, nheads, seqlen, headdim, device, dtype)

    with torch.inference_mode():
        res_pt_eager = model(attention_pytorch_eager, dropout_p=0.0)
        res_pt_native = model(attention_pytorch_native, dropout_p=0.0)

        assert torch.allclose(
            res_pt_eager, res_pt_native, atol=5e-3
        ), f" Maxdiff: {(res_pt_eager - res_pt_native).abs().max()}"

    time_pt_eager = benchmark_forward_backward(
        func=model,
        n_repeat=n_repeat,
        attention_func=attention_pytorch_eager,
    )

    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    time_pt_native = benchmark_forward_backward(
        func=model,
        n_repeat=n_repeat,
        attention_func=attention_pytorch_native,
    )

    print(f"PT eager: {time_pt_eager:.3f} ms")
    print(f"PT native: {time_pt_native:.3f} ms")

    output_file.write(
        f"{batch_size},{seqlen},{headdim},{nheads},{time_pt_eager:.3f},{time_pt_native:.3f},{time_pt_eager / time_pt_native:.3f}\n"
    )

output_file.close()
