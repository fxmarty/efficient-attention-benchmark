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


def benchmark_memory(module, device, train=True, **kwargs):
    torch.cuda.reset_max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    if train is False:
        module = module.eval()
        with torch.inference_mode():
            _ = module(**kwargs)
    else:
        module.zero_grad()
        res = module(**kwargs)
        loss = res.sum()
        loss.backward()
    torch.cuda.synchronize()

    max_memory = torch.cuda.max_memory_allocated(device)
    return max_memory


def benchmark_forward_backward(module, n_repeat: int, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_repeat):
        module.zero_grad()
        res = module(**kwargs)
        loss = res.sum()
        loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event)) / n_repeat


torch.manual_seed(0)
n_repeat = 30
dropout_p = 0.0
suffix = ""
causal = False
dtype = torch.float16  # torch.float32 is not supported for Hazy-Research implementation
device = "cuda"

# suffix = "gpt2"
# head_dim = [64]
# num_heads = [12]

suffix = "gpt-j-6B"
head_dim = [256]
num_heads = [16]

# suffix = "gpt-neox-20b"
# head_dim = [96]
# num_heads = [64]

all_parameters = {
    "batch_size": [8, 16, 64, 128, 256, 512],
    "seq_len": [64, 128, 256, 512, 1024],
    "head_dim": head_dim,
    "num_heads": num_heads,
}


def grid(
    parameters: Dict[str, Iterable[Any]],
) -> Iterable:
    for params in itertools.product(*parameters.values()):
        returned_list = list(params)
        yield returned_list


if suffix:
    suffix = f"_{suffix}"
output_file = open(f"benchmark_training_attention{suffix}.csv", "w")
output_file.write(
    "bs, seqlen, headdim, nheads, PT eager (ms/forward), PT native (ms/forward), Speedup, PT eager peak mem (MB), PT native peak mem (MB), Native mem saving\n"
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
        module=model,
        n_repeat=n_repeat,
        attention_func=attention_pytorch_eager,
    )

    max_memory_pt_eager = benchmark_memory(
        module=model,
        attention_func=attention_pytorch_eager,
        device=device,
    )

    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    time_pt_native = benchmark_forward_backward(
        module=model,
        n_repeat=n_repeat,
        attention_func=attention_pytorch_native,
    )

    max_memory_pt_native = benchmark_memory(
        module=model,
        attention_func=attention_pytorch_native,
        device=device,
    )

    max_memory_pt_eager = max_memory_pt_eager * 1e-6
    max_memory_pt_native = max_memory_pt_native * 1e-6

    print(f"PT eager: {time_pt_eager:.3f} ms, peak {max_memory_pt_eager:.2f} MB")
    print(f"PT native: {time_pt_native:.3f} ms, peak {max_memory_pt_native:.2f} MB")

    output_file.write(
        f"{batch_size},{seqlen},{headdim},{nheads},{time_pt_eager:.3f},{time_pt_native:.3f},{time_pt_eager / time_pt_native:.3f},{max_memory_pt_eager:.2f},{max_memory_pt_native:.2f},{max_memory_pt_eager / max_memory_pt_native:.3f}\n"
    )

output_file.close()
