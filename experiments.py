#!/usr/bin/env python
# coding: utf-8


from transformers import AutoTokenizer, AutoModelForCausalLM


from typing import Sequence
import torch
import numpy as np
from tqdm import tqdm
import copy


_SSM_NAME = "JackFram/llama-160m"
_LLM_NAME = "openlm-research/open_llama_3b_v2"
device = "cuda"

assert torch.cuda.is_available()
tokenizer = AutoTokenizer.from_pretrained(_SSM_NAME)
ssm = (
    AutoModelForCausalLM.from_pretrained(_SSM_NAME, torch_dtype=torch.float16)
    .cuda()
    .eval()
)
llm = (
    AutoModelForCausalLM.from_pretrained(
        _LLM_NAME,
        torch_dtype=torch.float16,
    )
    .cuda()
    .eval()
)


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
    if kv_cache_num_tokens > 0:
        attention_mask = torch.cat(
            (
                torch.ones(
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    kv_cache_num_tokens,
                    dtype=torch.float16,
                ).to(device),
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
                truly_unique_tokens.append(copy.copy(preceding_tokens)) # is this okay or do I need to make a copy?
                tokens_to_keep.append(tok)
                positions.append(pos)
                preceding_matrix_locs.append(len(tokens_to_keep) - 1)
                for preceding in preceding_matrix_locs:
                    mask_1[len(tokens_to_keep) - 1][preceding] = 1.0
            else:
                preceding_matrix_locs.append(truly_unique_tokens.index(preceding_tokens))
    
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
        device="cuda",
    )
    v = torch.rand(
        batch_size,
        num_attention_heads,
        kv_cache_num_tokens,
        hidden_size // num_attention_heads,
        dtype=torch.float16,
        device="cuda",
    )
    return tuple((k, v) for _ in range(num_layers))


def time_normal(input_ids, model: AutoModelForCausalLM, kv_cache=None):
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            past_key_values=kv_cache,
            use_cache=kv_cache is not None,
        )


def time_tree(
    input_ids, mask, position_ids, model: AutoModelForCausalLM, kv_cache=None
):
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=kv_cache is not None,
        )


from torch.profiler import profile, ProfilerActivity, schedule

# Guide: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

_N_ITERATIONS = 10
_WAIT_STEPS = 1
_WARMUP_STEPS = 1
schedule_params = {
    "wait": _WAIT_STEPS,
    "warmup": _WARMUP_STEPS,
    "active": _N_ITERATIONS - _WAIT_STEPS - _WARMUP_STEPS,
}
profiler_kwargs = {
    "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    "profile_memory": True,
    "schedule": schedule(**schedule_params),
    "record_shapes": True,
}


def print_normal_profile_stats(input, model):
    with torch.inference_mode(), profile(**profiler_kwargs) as prof:
        for _ in range(_N_ITERATIONS):
            model(
                input_ids=input, past_key_values=torch.tensor([3, 5, 7]), use_cache=True
            )
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


def print_tree_profile_stats(input, mask, position_ids, model):
    with torch.inference_mode(), profile(**profiler_kwargs) as prof:
        for _ in range(_N_ITERATIONS):
            model(
                input_ids=input,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=torch.tensor([1, 2, 3]),
                use_cache=True,
            )
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


import itertools
import gc


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


# NEW:
import torch.utils.benchmark as benchmark
import numpy as np


N_ITERATIONS = 32

expansion_configs = []
for length in range(2, 5):
    expansion_configs.extend(
        generate_expansion_configs(length, 1, 6)
    )  # first arg = length of config, second arg = min val in config, third = max
kv_sizes = [0, 2, 4, 8, 16, 64, 128]
# expansion_configs = [(7, 7, 7)]
# kv_sizes = [1024]
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
            sequential_measurement = sequential_timer.blocked_autorange(min_run_time=1)
            seq_time = sequential_measurement.mean
            print("Sequential Time: ", seq_time)

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
            mem_gb = end_memory_collection()
            utilization = torch.cuda.utilization()
            sequential_times[overall_conf] = [seq_time, mem_gb, utilization]
            print("Mem GB: ", mem_gb)
            print("Utilization: ", utilization)

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
            # print(correct_len)
            # print(tree_input.shape[-1])
            # print(tree_mask.shape[2])
            assert(tree_input.shape[-1] == correct_len)
            assert(tree_mask.shape[2] == correct_len and tree_mask.shape[3] == correct_len)
            # Required for 4D mask support in new HF
            tree_mask = _invert_4d_attention_mask(tree_mask, kv_size)

            tree_timer = benchmark.Timer(
                stmt="time_tree(input_ids, mask, position_ids, model, kv_cache)",
                setup="from __main__ import time_tree",
                num_threads=torch.get_num_threads(),
                globals={
                    "input_ids": tree_input,
                    "mask": tree_mask,
                    "position_ids": tree_position_ids,
                    "model": llm,
                    "kv_cache": kv_cache_tree,
                },
                label="Tree",
            )

            tree_measurement = tree_timer.blocked_autorange(min_run_time=1)
            tree_time = tree_measurement.mean
            print("Tree Time: ", tree_time)

            # Warmup
            time_tree(
                input_ids=tree_input,
                mask=tree_mask,
                position_ids=tree_position_ids,
                model=llm,
                kv_cache=kv_cache_tree,
            )
            reset_memory()
            time_tree(
                input_ids=tree_input,
                mask=tree_mask,
                position_ids=tree_position_ids,
                model=llm,
                kv_cache=kv_cache_tree,
            )
            mem_gb = end_memory_collection()
            utilization = torch.cuda.utilization()
            print("Mem GB: ", mem_gb)
            print("Utilization: ", utilization)
            tree_times[overall_conf] = [tree_time, mem_gb, utilization]
            reset_memory()
            print("-----------")
        except RuntimeError as e:
            print("SKIPPED")
            continue

# In[12]:


print(sequential_times)
print(tree_times)


# In[13]:


import pickle

f = open("saved_sequential.pkl", "wb")
pickle.dump(sequential_times, f)
f.close()
f2 = open("saved_tree.pkl", "wb")
pickle.dump(tree_times, f2)
f2.close()
