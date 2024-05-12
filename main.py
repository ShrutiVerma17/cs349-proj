from transformers import AutoTokenizer, AutoModelForCausalLM

from collections.abc import Sequence
import torch

_SSM_NAME = "JackFram/llama-160m"


def _create_token_tree(
    expansion_config: Sequence[int],
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
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


def main():
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(_SSM_NAME)
    ssm = AutoModelForCausalLM.from_pretrained(_SSM_NAME).cuda()

    token_tree = _create_token_tree(
        expansion_config=(2, 1, 2),
        prompt="The",
        tokenizer=tokenizer,
        model=ssm,
    )
    print(token_tree)


if __name__ == "__main__":
    main()
