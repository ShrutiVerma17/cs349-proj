import dataclasses
import torch


@dataclasses.dataclass
class GPU:
    tflops: float
    memory_bandwidth_tb_s: float


# https://www.nvidia.com/en-us/data-center/tesla-t4/
T4 = GPU(tflops=8.1, memory_bandwidth_tb_s=0.3)


def _total_tflops(
    token_batch: torch.Tensor,
    num_layers: int,
    d_model: int,
    n_head: int,
    vocab_size: int,
    kv_cache_token_count: int,
) -> float:
    # multihead attention flops - most important
    batch_size, seq_len = token_batch.size()
    mha_flops = (8 * (d_model**2) * seq_len * batch_size) + (
        (4 * d_model) + (5 * n_head)
    ) * batch_size * seq_len * (seq_len + kv_cache_token_count)

    # feedforward flops - less important
    ff_flops = 16 * batch_size * seq_len * d_model**2

    # LM head flops - even less important
    lm_head_flops = 2 * batch_size * seq_len * d_model * vocab_size

    return ((mha_flops + ff_flops) * num_layers + lm_head_flops) * 1e-12


def _memory_usage_tb(
    token_batch: torch.Tensor,
    dtype: torch.dtype,
    num_layers: int,
    d_model: int,
    n_head: int,
    vocab_size: int,
    kv_cache_token_count: int,
) -> float:
    batch_size, seq_len = token_batch.size()

    # multihead attention memory - most important
    mha_memory = (
        (4 * (d_model**2))
        + (12 * d_model * seq_len * batch_size)
        + (2 * batch_size * (seq_len + kv_cache_token_count) * d_model)
        + (6 * n_head * batch_size * seq_len * (seq_len + kv_cache_token_count))
        + seq_len * (seq_len + kv_cache_token_count)
    )

    # feedforward memory - less important
    ff_memory = 8 * d_model**2 + (10 * d_model * seq_len * batch_size)

    # LM head memory - even less important
    lm_head_memory = (
        (batch_size * seq_len * d_model)
        + (d_model * vocab_size)
        + (batch_size * seq_len * vocab_size)
    )

    element_size = dtype.itemsize
    return (
        element_size * (((mha_memory + ff_memory) * num_layers) + lm_head_memory)
    ) * 1e-12


def identify_compute_memory_bound(
    gpu: GPU,
    token_batch: torch.Tensor,
    dtype: torch.dtype,
    num_layers: int,
    d_model: int,
    n_head: int,
    vocab_size: int,
    kv_cache_token_count: int,
) -> None:
    tflops = _total_tflops(
        token_batch=token_batch,
        num_layers=num_layers,
        d_model=d_model,
        n_head=n_head,
        vocab_size=vocab_size,
        kv_cache_token_count=kv_cache_token_count,
    )
    memory_tb = _memory_usage_tb(
        token_batch=token_batch,
        dtype=dtype,
        num_layers=num_layers,
        d_model=d_model,
        n_head=n_head,
        vocab_size=vocab_size,
        kv_cache_token_count=kv_cache_token_count,
    )
    arithmetic_intensity = tflops / memory_tb
    gpu_ratio = gpu.tflops / gpu.memory_bandwidth_tb_s

    if arithmetic_intensity > gpu_ratio:
        print(
            f"Compute-bound: arithmetic intensity {arithmetic_intensity} > {gpu_ratio}"
        )
    else:
        print(
            f"Memory-bound: arithmetic intensity {arithmetic_intensity} < {gpu_ratio}"
        )
