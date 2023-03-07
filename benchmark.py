import itertools
import math
from typing import Any, Dict, Iterable

import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from tqdm import tqdm


def attention_pytorch_eager(query, key, value, is_causal=False, attn_mask=None, dropout_p=0.0):
    L = query.size(-1)
    S = value.size(-1)
    attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    if attn_mask is None:
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(L)), dim=-1)
    else:
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(L)) + attn_mask, dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def attention_pytorch_native(query, key, value, is_causal=False, attn_mask=None, dropout_p=0.0):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
    )


def hazy_preprocess(query, key, value, is_causal=False, attn_mask=None):
    batch_size = query.size(0)
    seqlen = query.size(2)

    # before: [batch_size, nheads, seqlen, headdim]
    query = torch.transpose(query, 1, 2).contiguous()
    key = torch.transpose(key, 1, 2).contiguous()
    value = torch.transpose(value, 1, 2).contiguous()

    # after: [batch_size * seqlen, nheads, headdim]
    query = query.view(-1, *query.shape[2:])
    key = key.view(-1, *key.shape[2:])
    value = value.view(-1, *value.shape[2:])

    qkv = torch.stack((query, key, value), dim=1)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)

    return qkv, cu_seqlens, seqlen


def attention_hazy_research(query, key, value, is_causal=False, attn_mask=None, dropout_p=0.0):
    batch_size = query.shape[0]
    qkv, cu_seqlens, seqlen = hazy_preprocess(query, key, value, is_causal, attn_mask)

    res = flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, seqlen, dropout_p=dropout_p, causal=is_causal)

    # before: [batch_size * seqlen, nheads, headdim]
    # after: [batch_size, nheads, seqlen, headdim]
    res = res.view(batch_size, seqlen, *res.shape[1:])
    res = res.transpose(1, 2)

    return res


def benchmark_forward(func, n_repeat: int, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_repeat):
        func(**kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event)) / n_repeat


def benchmark_hazy(n_repeat: int, query, key, value, is_causal=False, attn_mask=None, dropout_p=0.0):
    qkv, cu_seqlens, seqlen = hazy_preprocess(query, key, value, is_causal, attn_mask)

    time_hazy = benchmark_forward(
        flash_attn_unpadded_qkvpacked_func,
        n_repeat=n_repeat,
        qkv=qkv,
        cu_seqlens=cu_seqlens,
        max_seqlen=seqlen,
        dropout_p=0.0,
        causal=is_causal,
    )

    return time_hazy


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


output_file = open("benchmark_attention.csv", "w")
output_file.write(
    "batch_size, seq_len, headdim, nheads, PT eager (ms/forward), PT native (ms/forward), HazyResearch (ms/forward), Hazy speedup over scaled_dot_product_attention\n"
)

for params in tqdm(list(grid(all_parameters))):
    batch_size, seqlen, headdim, nheads = tuple(params)
    print(f"Running: bs={batch_size}, seqlen={seqlen}, headdim={headdim}, nheads={nheads}")

    qkv = torch.randn(batch_size, 3, nheads, seqlen, headdim, device=device, dtype=dtype)
    query, key, value = qkv.unbind(dim=1)

    with torch.inference_mode():
        res_pt_eager = attention_pytorch_eager(query, key, value)
        res_pt_native = attention_pytorch_native(query, key, value)
        res_hazy = attention_hazy_research(query, key, value)

        assert torch.allclose(
            res_pt_eager, res_pt_native, atol=5e-3
        ), f" Maxdiff: {(res_pt_eager - res_pt_native).abs().max()}"
        assert torch.allclose(res_pt_eager, res_hazy, atol=5e-3), f" Maxdiff: {(res_pt_eager - res_hazy).abs().max()}"

        time_pt_eager = benchmark_forward(
            attention_pytorch_eager,
            query=query,
            key=key,
            value=value,
            n_repeat=n_repeat,
            is_causal=False,
        )

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            time_pt_native = benchmark_forward(
                attention_pytorch_native,
                query=query,
                key=key,
                value=value,
                n_repeat=n_repeat,
                is_causal=False,
            )
        time_hazy = benchmark_hazy(query=query, key=key, value=value, n_repeat=n_repeat, is_causal=False)

        print(f"PT eager: {time_pt_eager:.3f} ms")
        print(f"PT native: {time_pt_native:.3f} ms")
        print(f"Hazy: {time_hazy:.3f} ms")

        output_file.write(
            f"{batch_size},{seqlen},{headdim},{nheads},{time_pt_eager:.3f},{time_pt_native:.3f},{time_hazy:.3f},{time_pt_native / time_hazy:.3f}\n"
        )

output_file.close()
