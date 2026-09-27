'''
Compare my model against the one from huggingface to make sure they produce the same logits
'''
import sys
from pathlib import Path

# Add the project root (JInfer) to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))


from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
from model.jllama import JLlama
from model.util import load_hf_weights

def basic():
    MODEL_ID = "meta-llama/Llama-3.1-8B"

    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenzier = AutoTokenizer.from_pretrained(MODEL_ID)

    hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=config.dtype,
    ).to(device="mps")

    personal_model = JLlama(config)
    personal_model = load_hf_weights(personal_model, hf_model)
    personal_model = personal_model.to(dtype=hf_model.dtype)
    personal_model = personal_model.to(device="mps")

    prompts = [
        "Hello my name is Julian",
        "Tell me all about apples",
        "Who is inside of the duolingo suit?",
        "Hables espanol?"
    ]

    mismatch_counter = 0
    max_diff_sum = 0.0
    mean_diff_sum = 0.0
    for prompt in prompts:
        tokens = tokenzier(prompt, return_tensors="pt").to("mps")
        with torch.no_grad():
            hf_out = hf_model(**tokens).logits.float()
            my_out = personal_model(tokens["input_ids"]).float()
        if not torch.allclose(hf_out, my_out):
            diff = (hf_out - my_out).abs()
            mismatch_counter += 1
            print(f"Mismatch {mismatch_counter}! hf_out:")
            print(hf_out)
            print("my_out:")
            print(my_out)
            max_diff_sum += diff.max().item()
            print(f"max absolute difference: {diff.max().item()}")
            mean_diff_sum += diff.mean().item()
            print(f"mean absolute difference: {diff.mean().item()}")

    print(f"Completed comparison, {mismatch_counter} mismatches. max_diff_avg = {max_diff_sum / mismatch_counter}, mean_diff_avg = {mean_diff_sum / mismatch_counter}")


if __name__ == "__main__":
    basic()
