import torch

a = torch.load("original_attn_output.pt")
b = torch.load("ring_attn_output.pt")

abs_delta = torch.abs(a - b)

print(f"total delta: {abs_delta.sum().item()}, mean: {abs_delta.mean().item()}")

delta_per_row = abs_delta.mean(dim=-1)
print("delta_per_row:", delta_per_row)
