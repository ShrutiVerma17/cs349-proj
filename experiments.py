#!/usr/bin/env python
# coding: utf-8


from transformers import AutoTokenizer, AutoModelForCausalLM


from typing import Sequence
import torch
import numpy as np
from tqdm import tqdm
import copy
import argparse
import itertools
import gc
import torch.utils.benchmark as benchmark
import numpy as np
import pickle
from datetime import datetime
import dataclasses

# from torch.nn.attention import SDPBackend, sdpa_kernel


_SSM_NAME = "JackFram/llama-160m"
_LLM_NAME = "openlm-research/open_llama_3b_v2"
device = "cuda"

assert torch.cuda.is_available()
tokenizer = AutoTokenizer.from_pretrained(_SSM_NAME)


_PROMPT = "The good dog is"
assert len(tokenizer.tokenize(_PROMPT)) == 4


def _create_token_tree(
    expansion_config: Sequence[int],
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    has_kv_cache: bool = False,
):
    """Create token tree following Figure 3 in the paper.

    We don't need "real" tokens for our experiments - just
    random integers would work too - but might as well.

    Figure 3 illustrates the <k1, k2, ...> expansion approach they
    use to create token trees. We can use each of the top_k tokens from
    a single model to create the same tree structure.

    Args:
        expansion_config: A sequence of integers representing how much to
            branch at each generation step.
        prompt: Initial prompt.
        tokenizer: HF tokenizer.
        model: HF generative model.
    """
    assert expansion_config
    current_tree = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    if has_kv_cache:
        assert tokenizer.add_bos_token
        current_tree = current_tree[:, 1:]
    else:
        current_tree = current_tree[:, :-1]
    assert current_tree.shape[-1] == 4
    for k in expansion_config:
        output = model.generate(
            current_tree,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # Take the top_k tokens from the 1 generation step we've done
        top_k = torch.topk(output.scores[-1], k=k, dim=-1).indices.reshape(-1, 1)
        current_tree = torch.repeat_interleave(current_tree, k, dim=0)
        # Join the top_k tokens to the current tree
        current_tree = torch.cat((current_tree, top_k), dim=-1)

    return current_tree


def _invert_4d_attention_mask(
    attention_mask: torch.Tensor, kv_cache_num_tokens: int = 0
) -> torch.Tensor:
    """For 4D masks, new HF requires us to invert the mask so it doesn't modify it at all."""
    # The attention mask must have last 2 dims shape [current seq len, KV cache size + current seq len]
    # So we prepend a tensor of 1s to allow attending to the full KV cache
    assert attention_mask.dim() == 4
    attention_mask = torch.cat(
        (
            torch.ones(
                attention_mask.shape[0],
                attention_mask.shape[1],
                attention_mask.shape[2],
                kv_cache_num_tokens,
                dtype=torch.float16,
                device=device,
            ),
            attention_mask,
        ),
        dim=-1,
    )
    # Invert the mask: 0s to -inf and 1s to 0 (0 means attention allowed)
    min_dtype = torch.finfo(torch.float16).min
    min_dtype = min_dtype if attention_mask.dtype == torch.float16 else -1e4
    attention_mask.masked_fill_(attention_mask == 0.0, min_dtype)
    attention_mask.masked_fill_(attention_mask == 1.0, 0.0)
    return attention_mask


def construct_tree_model_inputs(sequences):
    flat = torch.flatten(sequences).tolist()
    truly_unique_tokens = []
    tokens_to_keep = []
    positions = []
    mask_1 = np.zeros((len(flat), len(flat)))

    for seq_num, seq in enumerate(sequences):
        preceding_tokens = []
        preceding_matrix_locs = []
        for pos, tok in enumerate(seq):
            preceding_tokens.append(tok)
            if preceding_tokens not in truly_unique_tokens:
                truly_unique_tokens.append(
                    copy.copy(preceding_tokens)
                )  # is this okay or do I need to make a copy?
                tokens_to_keep.append(tok)
                positions.append(pos)
                preceding_matrix_locs.append(len(tokens_to_keep) - 1)
                for preceding in preceding_matrix_locs:
                    mask_1[len(tokens_to_keep) - 1][preceding] = 1.0
            else:
                preceding_matrix_locs.append(
                    truly_unique_tokens.index(preceding_tokens)
                )

    input_1 = torch.tensor([tokens_to_keep]).to(device)

    a = input_1.shape[-1]
    mask_1 = mask_1[:a, :a]

    mask_1 = torch.tensor(mask_1, device=device, dtype=torch.float16)
    mask_1 = mask_1.unsqueeze(0).unsqueeze(0).to(device)
    position_ids_1 = torch.tensor([positions], device=device, dtype=torch.int)
    return (input_1, mask_1, position_ids_1)


def _create_dummy_kv_cache(
    kv_cache_num_tokens: int,
    batch_size: int,
    num_attention_heads: int,
    hidden_size: int,
    num_layers: int,
):
    k = torch.rand(
        batch_size,
        num_attention_heads,
        kv_cache_num_tokens,
        hidden_size // num_attention_heads,
        dtype=torch.float16,
        device=device,
    )
    v = torch.rand(
        batch_size,
        num_attention_heads,
        kv_cache_num_tokens,
        hidden_size // num_attention_heads,
        dtype=torch.float16,
        device=device,
    )
    return tuple((k, v) for _ in range(num_layers))


def time_normal(input_ids, model: AutoModelForCausalLM, kv_cache=None):
    # with torch.inference_mode(), sdpa_kernel(
    #     [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
    # ):
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            past_key_values=kv_cache,
            use_cache=kv_cache is not None,
        )


def time_tree(
    input_ids,
    mask,
    position_ids,
    model: AutoModelForCausalLM,
    kv_size,
    kv_cache=None,
):
    # with torch.inference_mode(), sdpa_kernel(
    #     [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
    # ):
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            # Required for 4D mask support in new HF. Include in timing since it's included for sequential in HF code.
            attention_mask=_invert_4d_attention_mask(mask, kv_size),
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=kv_cache is not None,
        )


def generate_expansion_configs(config_length, min_expansion, max_expansion):
    values = [i for i in range(min_expansion, max_expansion)]
    result = list(itertools.product(values, repeat=config_length))
    return result


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()


def end_memory_collection():
    torch.cuda.synchronize()
    max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    return max_mem_gb


@dataclasses.dataclass
class BenchmarkResult:
    mean_time: float
    median_time: float
    p90_time: float
    num_samples: int
    mem_gb: float
    utilization: float


def main(parser):
    args = parser.parse_args()

    ssm = (
        AutoModelForCausalLM.from_pretrained(_SSM_NAME, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )

    attn_implementation = "flash_attention_2" if args.flash else "sdpa"
    llm = (
        AutoModelForCausalLM.from_pretrained(
            _LLM_NAME,
            attn_implementation=attn_implementation,
            torch_dtype=torch.float16,
        )
        .cuda()
        .eval()
    )

    expansion_configs = []
    for length in range(2, 5):
        expansion_configs.extend(
            generate_expansion_configs(length, 1, 6)
        )  # first arg = length of config, second arg = min val in config, third = max
    # expansion_configs = [[32, 2, 2] + [1 for _ in range(29)]]
    kv_sizes = [0, 2, 4, 8, 16, 64, 128]
    # past_key_values, need tuple of two tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head))
    sequential_times = {}
    tree_times = {}

    for config in tqdm(expansion_configs, desc="Configs"):
        for kv_size in tqdm(kv_sizes, "KV cache sizes"):
            print("-----------")
            overall_conf = str(config) + ", " + str(kv_size)
            print(overall_conf)
            try:
                token_tree = _create_token_tree(
                    expansion_config=config,
                    prompt=_PROMPT,
                    tokenizer=tokenizer,
                    model=ssm,
                    has_kv_cache=kv_size > 0,
                )

                batch_size = np.prod(config)
                kv_cache_sequential = _create_dummy_kv_cache(
                    kv_cache_num_tokens=kv_size,
                    batch_size=batch_size,
                    num_attention_heads=llm.config.num_attention_heads,
                    hidden_size=llm.config.hidden_size,
                    num_layers=llm.config.num_hidden_layers,
                )

                if batch_size % 8 == 0 and token_tree.shape[-1] % 8 == 0:
                    print("Multiples of 8!")

                sequential_timer = benchmark.Timer(
                    stmt="time_normal(input_ids, model, kv_cache)",
                    setup="from __main__ import time_normal",
                    num_threads=torch.get_num_threads(),
                    globals={
                        "input_ids": token_tree,
                        "model": llm,
                        "kv_cache": kv_cache_sequential,
                    },
                    label="Sequential",
                )
                sequential_measurement = sequential_timer.blocked_autorange(
                    min_run_time=args.min_run_time
                )
                seq_time_mean = sequential_measurement.mean
                seq_time_median = sequential_measurement.median
                seq_num_samples = len(sequential_measurement.times)
                seq_p90 = np.percentile(sequential_measurement.times, 90)
                print(
                    f"Sequential Time\nNum Samples: {seq_num_samples} Median: {seq_time_median} P90: {seq_p90} Mean: {seq_time_mean}"
                )

                # Warmup
                time_normal(
                    input_ids=token_tree,
                    model=llm,
                    kv_cache=kv_cache_sequential,
                )
                reset_memory()
                time_normal(
                    input_ids=token_tree,
                    model=llm,
                    kv_cache=kv_cache_sequential,
                )
                utilization = torch.cuda.utilization()
                mem_gb = end_memory_collection()
                sequential_times[overall_conf] = BenchmarkResult(
                    mean_time=seq_time_mean,
                    median_time=seq_time_median,
                    p90_time=seq_p90,
                    num_samples=seq_num_samples,
                    mem_gb=mem_gb,
                    utilization=utilization,
                )
                print("Mem GB: ", mem_gb)
                print("Utilization: ", utilization)

                if args.flash:
                    # Skipping tree for flash attention since it can't use it
                    continue

                # construct inputs for tree decoding
                kv_cache_tree = _create_dummy_kv_cache(
                    kv_cache_num_tokens=kv_size,
                    batch_size=1,
                    num_attention_heads=llm.config.num_attention_heads,
                    hidden_size=llm.config.hidden_size,
                    num_layers=llm.config.num_hidden_layers,
                )
                tree_input, tree_mask, tree_position_ids = construct_tree_model_inputs(
                    token_tree
                )
                correct_len = len(_PROMPT.split(" ")) + np.sum(np.cumprod(config))
                assert tree_input.shape[-1] == correct_len
                assert (
                    tree_mask.shape[2] == correct_len
                    and tree_mask.shape[3] == correct_len
                )

                tree_timer = benchmark.Timer(
                    stmt="time_tree(input_ids, mask, position_ids, model, kv_size, kv_cache)",
                    setup="from __main__ import time_tree",
                    num_threads=torch.get_num_threads(),
                    globals={
                        "input_ids": tree_input,
                        "mask": tree_mask,
                        "position_ids": tree_position_ids,
                        "model": llm,
                        "kv_size": kv_size,
                        "kv_cache": kv_cache_tree,
                    },
                    label="Tree",
                )

                tree_measurement = tree_timer.blocked_autorange(
                    min_run_time=args.min_run_time
                )
                tree_time_mean = tree_measurement.mean
                tree_time_median = tree_measurement.median
                tree_num_samples = len(tree_measurement.times)
                tree_p90 = np.percentile(tree_measurement.times, 90)
                print(
                    f"Tree Time\nNum Samples: {tree_num_samples} Median: {tree_time_median} P90: {tree_p90} Mean: {tree_time_mean}"
                )

                # Warmup
                time_tree(
                    input_ids=tree_input,
                    mask=tree_mask,
                    position_ids=tree_position_ids,
                    model=llm,
                    kv_size=kv_size,
                    kv_cache=kv_cache_tree,
                )
                reset_memory()
                time_tree(
                    input_ids=tree_input,
                    mask=tree_mask,
                    position_ids=tree_position_ids,
                    model=llm,
                    kv_size=kv_size,
                    kv_cache=kv_cache_tree,
                )
                mem_gb = end_memory_collection()
                utilization = torch.cuda.utilization()
                print("Mem GB: ", mem_gb)
                print("Utilization: ", utilization)
                tree_times[overall_conf] = BenchmarkResult(
                    mean_time=tree_time_mean,
                    median_time=tree_time_median,
                    p90_time=tree_p90,
                    num_samples=tree_num_samples,
                    mem_gb=mem_gb,
                    utilization=utilization,
                )
                reset_memory()
                print("-----------")
            except RuntimeError as e:
                print("SKIPPED")
                continue

    print(sequential_times)
    print(tree_times)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    sequential_file_name = f"saved_sequential"
    if args.flash:
        sequential_file_name += "_flash"
    f = open(f"{sequential_file_name}_{timestamp}.pkl", "wb")
    pickle.dump(sequential_times, f)
    f.close()
    f2 = open(f"saved_tree_{timestamp}.pkl", "wb")
    pickle.dump(tree_times, f2)
    f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_run_time",
        type=int,
        default=1,
        help="Minimum number of seconds to run the benchmark (blocked_autorange parameter).",
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Allow flash attention 2 for sequential decoding. Skips tree since this would throw an error.",
    )
    main(parser)
