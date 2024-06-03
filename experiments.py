#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[1]:


import transformers
print(transformers.__version__)


# In[3]:


from transformers import AutoTokenizer, AutoModelForCausalLM

# from collections.abc import Sequence
from typing import Sequence
import torch
import numpy as np
import time

_SSM_NAME = "JackFram/llama-160m"
_LLM_NAME = 'openlm-research/open_llama_3b_v2'
device = "cuda"

assert torch.cuda.is_available()
tokenizer = AutoTokenizer.from_pretrained(_SSM_NAME)
ssm = AutoModelForCausalLM.from_pretrained(_SSM_NAME, torch_dtype=torch.float16, device_map='auto')
llm = AutoModelForCausalLM.from_pretrained(_LLM_NAME, torch_dtype=torch.float16, device_map='auto')


# In[4]:


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


# In[5]:


def _invert_4d_attention_mask(attention_mask: torch.Tensor, kv_cache_num_tokens: int=0) -> torch.Tensor:
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
    min_dtype = min_dtype if attention_mask.dtype == torch.float16 else -1e+4
    attention_mask.masked_fill_(attention_mask == 0.0, min_dtype)
    attention_mask.masked_fill_(attention_mask == 1.0, 0.0)
    return attention_mask

def construct_tree_model_inputs(sequences):
    # input_1 = torch.unique(torch.flatten(sequences), sorted=False)
    flat = torch.flatten(sequences).tolist()
    unique = []
    for tok in flat:
        if tok not in unique:
            unique.append(tok)
    # input is list of unique tokens
    input_1 = torch.tensor([unique]).to(device)

    a = input_1.shape[-1]
    mask_1 = np.zeros((a, a))
    positions = [-1] * len(unique)
    
    for seq in sequences:
        branch_progress = []
        for (pos, tok) in enumerate(seq):
            input_1_idx = unique.index(tok)
            positions[input_1_idx] = pos
            branch_progress.append(input_1_idx)
            for idx in branch_progress:
                mask_1[input_1_idx][idx] = 1.0
    mask_1 = torch.tensor(mask_1, device=device, dtype=torch.float16)
    mask_1 = mask_1.unsqueeze(0).unsqueeze(0).to(device)
    position_ids_1 = torch.tensor([positions], device=device, dtype=torch.int)
    return (input_1, mask_1, position_ids_1)


# In[6]:


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


# In[7]:


def time_normal(input_ids, model: AutoModelForCausalLM):
    with torch.inference_mode():
        model(input_ids=input_ids)

def time_tree(input_ids, mask, position_ids, model: AutoModelForCausalLM):
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids)


# In[8]:


from torch.profiler import profile, ProfilerActivity, schedule

# Guide: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

_N_ITERATIONS = 10
_WAIT_STEPS = 1
_WARMUP_STEPS = 1
schedule_params = {
    'wait': _WAIT_STEPS,
    'warmup': _WARMUP_STEPS,
    'active': _N_ITERATIONS - _WAIT_STEPS - _WARMUP_STEPS,
}
profiler_kwargs = {
    'activities': [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    'profile_memory': True,
    'schedule': schedule(**schedule_params),
    'record_shapes': True
}

def print_normal_profile_stats(input, model):
    with torch.inference_mode(), profile(**profiler_kwargs) as prof:
        for _ in range(_N_ITERATIONS):
            model(input_ids=input, past_key_values = torch.tensor([3, 5, 7]), use_cache = True)
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

def print_tree_profile_stats(input, mask, position_ids, model):
    with torch.inference_mode(), profile(**profiler_kwargs) as prof:
        for _ in range(_N_ITERATIONS):
            model(input_ids=input, attention_mask=mask, position_ids=position_ids, past_key_values = torch.tensor([1, 2, 3]), use_cache = True)
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


# In[15]:


import itertools 
import gc
def generate_expansion_configs(config_length, min_expansion, max_expansion):
    values = [i for i in range(min_expansion, max_expansion)]
    result = list(itertools.product(values, repeat=config_length))
    return result

def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# In[ ]:


# NEW: 
import torch.utils.benchmark as benchmark
import numpy as np


N_ITERATIONS = 32

#tree_lengths = range(1, 10)
# B = range(2, 10)
# sequential_times = []
# tree_times = []

expansion_configs = generate_expansion_configs(3, 2, 8) # first arg = length of config, second arg = min val in config, third = max
kv_sizes = [0, 128, 256, 512, 1024, 2048]
# expansion_configs = [(7, 7, 7)]
# kv_sizes = [1024]

# past_key_values, need tuple of two tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head))
sequential_times = {}
tree_times = {}

for config in expansion_configs: 
    for kv_size in kv_sizes:
        print("-----------")
        overall_conf = str(config) + ", " + str(kv_size)
        print(overall_conf)
        token_tree = _create_token_tree(
                expansion_config=config,
                prompt="The",
                tokenizer=tokenizer,
                model=ssm,
            )

        batch_size=np.prod(config)
        try: 
            kv_cache_sequential = _create_dummy_kv_cache(
                kv_cache_num_tokens=kv_size,
                batch_size=batch_size,
                num_attention_heads=llm.config.num_attention_heads,
                hidden_size=llm.config.hidden_size,
                num_layers=llm.config.num_hidden_layers
            )
        except:
            print("SKIPPED")
            continue
        
        sequential_timer = benchmark.Timer(
            stmt="time_normal(input_ids, model)",
            setup="from __main__ import time_normal",
            num_threads=torch.get_num_threads(),
            globals={
                'input_ids': token_tree,
                'model': llm,
                'kv_cache': kv_cache_sequential
            },
            label="Sequential"
        )
        sequential_measurement = sequential_timer.timeit(N_ITERATIONS)
        seq_time = sequential_measurement.times[-1]
        sequential_times[overall_conf] = seq_time
        print("Sequential Time: ", seq_time)
        
        # construct inputs for tree decoding
        kv_cache_tree = _create_dummy_kv_cache(
            kv_cache_num_tokens=kv_size,
            batch_size=1,
            num_attention_heads=llm.config.num_attention_heads,
            hidden_size=llm.config.hidden_size,
            num_layers=llm.config.num_hidden_layers
        )
        tree_input, tree_mask, tree_position_ids = construct_tree_model_inputs(token_tree)
        # Required for 4D mask support in new HF
        tree_mask = _invert_4d_attention_mask(tree_mask, kv_size)
        print(tree_input.is_cuda & tree_mask.is_cuda & tree_position_ids.is_cuda)
        tree_timer = benchmark.Timer(
            stmt="time_tree(input_ids, mask, position_ids, model)",
            setup="from __main__ import time_tree",
            num_threads=torch.get_num_threads(),
            globals={
                'input_ids': tree_input,
                'mask': tree_mask,
                'position_ids': tree_position_ids,
                'model': llm,
                'kv_cache': kv_cache_tree
            },
            label="Tree"
        )
        tree_position_ids = tree_position_ids.cpu()
        tree_measurement = tree_timer.timeit(N_ITERATIONS)
        tree_time = tree_measurement.times[-1]
        tree_times[overall_conf] = tree_time
        print("Tree Time: ", tree_time)
        #token_tree = token_tree.cpu()
        #tree_input = tree_input.cpu()
        #tree_mask = tree_mask.cpu()
        reset_memory()
        break
        print("-----------")
    break
time.sleep(5)

# In[12]:


print(sequential_times)
print(tree_times)


# In[13]:


import pickle 

f = open('saved_sequential.pkl', 'wb')
pickle.dump(sequential_times, f)
f.close()
f2 = open('saved_tree.pkl', 'wb')
pickle.dump(tree_times, f2)
f2.close()

#with open('saved_sequential.pkl', 'wb') as f:
#    pickle.dump(sequential_times, f)
#with open('saved_tree.pkl', 'wb') as f:
#    pickle.dump(tree_times, f)

