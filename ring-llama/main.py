
import os
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from modeling_llama import LlamaForCausalLM
from tokenization_llama_fast import LlamaTokenizerFast
import torch.distributed as dist


def load_model(model_name: str, cache_dir: str, torch_dtype: torch.dtype):
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
    model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    return model, tokenizer


def main():
    dtype = torch.float16
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:0")

    print(f"world_size: {world_size}, device: {device}")

    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf", cache_dir='/workspace/hf_home/', torch_dtype=dtype)
    model.eval()
    model.to(device)

    x = tokenizer("Hello I am the llama, ", return_tensors="pt")

    tokenized_input = x.input_ids
    #print("tokenized_input", tokenized_input.shape)

    input_chunks = tokenized_input.chunk(chunks=world_size, dim=1)
    #print("input_chunks", input_chunks)
    x = input_chunks[rank]

    print(f"model input x for rank: {rank}:", x)
    x = x.to(device)

    y = model(x).logits

    print(f"output logits for rank: {rank}:", y.shape, y.dtype, y.device)


if __name__ == '__main__':
    main()

