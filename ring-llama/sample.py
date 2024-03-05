import torch
import torch.nn.functional as F

"""
Contains various sampling strategies for logits

"""


def top_p_sampling(logits, p=0.9):
    """Applies top-p (nucleus) sampling to logits."""
    # Sort logits in descending order and compute their probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probabilities = torch.nn.functional.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Remove tokens with a cumulative probability above the threshold p
    indices_to_remove = cumulative_probabilities > p
    # Shift the indices to the right to keep the first token above p
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = False

    # Set the logits for the removed indices to negative infinity
    sorted_indices_to_remove = sorted_indices[indices_to_remove]
    logits[sorted_indices_to_remove] = float("-inf")

    return logits


def sample_from_logits(gathered_logits, strategy="top-k", k=5, p=0.9):
    concatenated_logits = torch.cat(gathered_logits, dim=1)
    sampled_indices = None

    if strategy == "greedy" or strategy == "top-k":
        probabilities = F.softmax(concatenated_logits, dim=-1)
        if strategy == "greedy":
            # Greedy sampling: select the token with the highest probability at each step
            sampled_indices = torch.argmax(probabilities, dim=-1)
        elif strategy == "top-k":
            probabilities = F.softmax(concatenated_logits, dim=-1)
            topk_vals, topk_indices = torch.topk(probabilities, k=k, dim=-1)
            # Ensuring topk_vals is 2D: [batch_size, k]
            if topk_vals.dim() > 2:
                topk_vals = topk_vals.view(-1, k)  # Reshape for safety, though it should already be [batch_size, k]

            # Sampling from the top-k values for each item in the batch
            # topk_vals is now guaranteed to be [batch_size, k], suitable for torch.multinomial
            sampled_from_topk = torch.multinomial(topk_vals, 1)  # [batch_size, 1], samples one index per batch item

            # Gathering the actual token indices corresponding to the sampled positions
            # Use torch.gather or advanced indexing to map back to original token indices
            batch_size = topk_indices.size(0)
            batch_indices = torch.arange(batch_size).unsqueeze(-1).to(topk_indices.device)
            sampled_indices = topk_indices[batch_indices, sampled_from_topk].squeeze(-1)  # Remove singleton dimension
    elif strategy == "top-p":
        # Apply top-p sampling to logits and then sample
        sampled_indices = torch.empty(
            concatenated_logits.size(0),
            concatenated_logits.size(1),
            dtype=torch.long,
            device=concatenated_logits.device,
        )
        for i in range(concatenated_logits.shape[1]):  # Iterate through sequence
            logits = concatenated_logits[:, i, :]
            filtered_logits = top_p_sampling(logits, p=p)
            probs = F.softmax(filtered_logits, dim=-1)
            # Use torch.multinomial to sample from the filtered distribution
            next_token_samples = torch.multinomial(
                probs, 1
            )  # Sample 1 token per sequence
            sampled_indices[:, i] = next_token_samples.squeeze(-1)

    return sampled_indices
