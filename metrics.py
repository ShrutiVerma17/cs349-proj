import torch


def total_tflops(
    token_batch: torch.Tensor,
    num_layers: int,
    d_model: int,
    n_head: int,
    vocab_size: int,
) -> int:
    # multihead attention flops - most important
    batch_size, seq_len = token_batch.size()
    mha_flops = (8 * (d_model**2) * seq_len * batch_size) + (
        (4 * d_model) + (5 * n_head)
    ) * batch_size * seq_len**2

    # feedforward flops - less important
    ff_flops = 16 * batch_size * seq_len * d_model**2

    # LM head flops - even less important
    lm_head_flops = 2 * batch_size * seq_len * d_model * vocab_size

    return ((mha_flops + ff_flops) * num_layers + lm_head_flops) * 1e-12


def memory_usage_tb(
    token_batch: torch.Tensor,
    num_layers: int,
    dtype: torch.dtype,
    d_model: int,
    n_head: int,
    vocab_size: int,
) -> int:
    batch_size, seq_len = token_batch.size()

    # multihead attention memory - most important
    mha_memory = (
        (4 * (d_model**2))
        + (12 * d_model * seq_len * batch_size)
        + (6 * n_head * batch_size * seq_len**2)
        + seq_len**2
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
