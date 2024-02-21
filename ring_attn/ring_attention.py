"""This code is translated to pytorch from the original at github.com/lhao499/RingAttention
For more details, refer to 'RingAttention' at https://arxiv.org/abs/2310.01889 and 'Blockwise Parallel Transformers' at https://arxiv.org/abs/2305.19370. 
"""

import numpy as np
from einops import rearrange
from functools import partial
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Dropout

import dataclasses
import functools
from typing import Any, NamedTuple


## Ring Attention
def _ring_attention_fwd(
    q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs
):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = torch.zeros((batch, q_len, num_heads, dim_per_head), dtype=q.dtype)
    denominator = torch.zeros((batch, num_heads, q_len), dtype=q.dtype)
    axis_size = dist.get_world_size(axis_name)
    q_block_size, kv_block_size = (
        q_len,
        kv_len,
    )  # assumes this function is pre-sharded inside shard_map
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        # torch.distributed.get_rank(group=None)
        # REVIEW this axis_name compatible?
        q_block_idx = dist.get_rank(group=axis_name)
        k_block_idx = (dist.get_rank(group=axis) - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        numerator, denominator, max_score = _blockwise_attention_fwd(
            q,
            k,
            v,
            (numerator, denominator, prev_max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=attn_bias,
            segment_ids=segment_ids,
            **blockwise_kwargs
        )
        k, v = map(lambda x: dist.scatter(x, dim=0, src=0), (k, v))
        return (max_score, numerator, denominator, k, v), None

    prev_max_score = torch.full((batch, num_heads, q_len), float("-inf")).to(q.dtype)
    (max_score, numerator, denominator, _, _), _ = torch.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v),
        xs=torch.arange(0, axis_size),
    )
    output = numerator / rearrange(denominator, "b h q -> b q h")[..., None]
    return output.to(v.dtype), (
        output,
        q,
        k,
        v,
        attn_bias,
        segment_ids,
        denominator,
        max_score,
    )


def _ring_attention_bwd(axis_name, float32_logits, blockwise_kwargs, res, g):
    del float32_logits
    output, q, k, v, attn_bias, segment_ids, denominator, max_score = res
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    axis_size = dist.get_world_size(axis_name)
    dq = torch.zeros_like(q, dtype=q.dtype)
    dk = torch.zeros_like(k, dtype=k.dtype)
    dv = torch.zeros_like(v, dtype=v.dtype)
    query_chunk_size = blockwise_kwargs["query_chunk_size"]
    key_chunk_size = blockwise_kwargs["key_chunk_size"]
    q_block_size, kv_block_size = (
        q_len,
        kv_len,
    )  # assumes this function is pre-sharded inside shard_map

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        q_block_idx = dist.get_rank()
        k_block_idx = (dist.get_rank() - idx) % axis_size
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(
            q,
            k,
            v,
            g,
            (dq, dk, dv, output, denominator, max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=attn_bias,
            segment_ids=segment_ids,
            **blockwise_kwargs
        )
        k, v, dk, dv = map(lambda x: dist.scatter(x, dim=0, src=0), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None

    (dq, dk, dv, k, v), _ = torch.scan(
        scan_kv_block, init=(dq, dk, dv, k, v), xs=torch.arange(0, axis_size)
    )
    dq, dk, dv = dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
    return dq, dk, dv, None, None


class RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_bias,
        segment_ids,
        axis_name,
        float32_logits,
        blockwise_kwargs,
    ):
        y, _ = _ring_attention_fwd(
            q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs
        )
        ctx.save_for_backward(
            q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (
            q,
            k,
            v,
            attn_bias,
            segment_ids,
            axis_name,
            float32_logits,
            blockwise_kwargs,
        ) = ctx.saved_tensors
        grad_q, grad_k, grad_v = _ring_attention_bwd(
            axis_name,
            float32_logits,
            blockwise_kwargs,
            (y, q, k, v, attn_bias, segment_ids, denominator, max_score),
            grad_output,
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None


ring_attention = RingAttention.apply


## Standard Ring Attention
def _ring_attention_standard_fwd(q, k, v, attn_mask, axis_name, float32_logits):
    if float32_logits:
        q, k = q.float(), k.float()
    batch, q_len, num_heads, _ = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = torch.zeros((batch, q_len, num_heads, dim_per_head), dtype=q.dtype)
    denominator = torch.zeros((batch, num_heads, q_len), dtype=q.dtype)
    axis_size = dist.get_world_size(axis_name)
    scale = torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        mask = attn_mask[
            ((dist.get_rank() - idx) % axis_size)
            * kv_len : ((dist.get_rank() - idx) % axis_size + 1)
            * kv_len
        ]
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / scale
        attn_weights = torch.where(
            mask, attn_weights, torch.finfo(attn_weights.dtype).min
        )
        max_score = torch.maximum(prev_max_score, torch.max(attn_weights, dim=-1))
        exp_weights = torch.exp(attn_weights - max_score[..., None])
        correction = rearrange(torch.exp(prev_max_score - max_score), "b h q -> b q h")[
            ..., None
        ]
        numerator = numerator * correction + torch.einsum(
            "bhqk,bkhd->bqhd", exp_weights, v
        )
        denominator = denominator * torch.exp(prev_max_score - max_score) + torch.sum(
            exp_weights, dim=-1
        )
        k, v = map(lambda x: dist.scatter(x, dim=0, src=0), (k, v))
        return (max_score, numerator, denominator, k, v), None

    prev_max_score = torch.full((batch, num_heads, q_len), float("-inf"), dtype=q.dtype)
    (max_score, numerator, denominator, _, _), _ = torch.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, k, v),
        xs=torch.arange(0, axis_size),
    )
    output = numerator / rearrange(denominator, "b h q -> b q h")[..., None]
    return output.to(v.dtype), (
        output,
        q,
        k,
        v,
        attn_mask,
        numerator,
        denominator,
        max_score,
    )


def _ring_attention_standard_bwd(axis_name, float32_logits, res, g):
    del float32_logits
    axis_size = dist.get_world_size(axis_name)
    output, q, k, v, attn_mask, numerator, denominator, max_score = res
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    batch, kv_len, num_heads, dim_per_head = k.shape
    scale = torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        mask = attn_mask[
            ((dist.get_rank() - idx) % axis_size)
            * kv_len : ((dist.get_rank() - idx) % axis_size + 1)
            * kv_len
        ]
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / scale
        attn_weights = torch.where(
            mask, attn_weights, torch.finfo(attn_weights.dtype).min
        )
        exp_weights = (
            torch.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        )
        ds = torch.einsum("bqhd,bkhd->bhqk", g, v)
        dl = (ds - torch.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        dq = dq + torch.einsum("bhqk,bkhd->bqhd", dl, k) / scale
        dk = dk + torch.einsum("bqhd,bhqk->bkhd", q, dl) / scale
        dv = dv + torch.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        k, v, dk, dv = map(lambda x: dist.scatter(x, dim=0, src=0), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None

    (dq, dk, dv, k, v), _ = torch.scan(
        scan_kv_block, init=(dq, dk, dv, k, v), xs=torch.arange(0, axis_size)
    )
    dq, dk, dv = dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
    return dq, dk, dv, None


class RingAttentionStandard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask, axis_name, float32_logits=True):
        y, _ = _ring_attention_standard_fwd(
            q, k, v, attn_mask, axis_name, float32_logits
        )
        ctx.save_for_backward(q, k, v, attn_mask, axis_name, float32_logits)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attn_mask, axis_name, float32_logits = ctx.saved_tensors
        grad_q, grad_k, grad_v = _ring_attention_standard_bwd(
            axis_name, float32_logits, (y, q, k, v, attn_mask, _, _, _), grad_output
        )
        return grad_q, grad_k, grad_v, None, None, None


ring_attention_standard = RingAttentionStandard.apply


def _blockwise_attention_fwd(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    bias,
    segment_ids,
    causal,
    query_chunk_size,
    key_chunk_size,
    deterministic,
    dropout_rng,
    attn_pdrop,
    dtype,
    policy,
    precision,
    prevent_cse,
):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: x.transpose(0, 1), (q, k, v))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape(
        (batch, num_q, query_chunk_size, num_heads, dim_per_head)
    )
    numerator = torch.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(
        lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score)
    )

    scale = torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = torch.split(dropout_rng, 2)
        attn_dropout = torch.bernoulli(attn_dropout_rng, attn_pdrop).expand(
            batch, num_heads, q_len, kv_len
        )
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size,
        key_chunk_size,
        bias,
        segment_ids,
        deterministic,
        attn_dropout,
        attn_pdrop,
        causal,
        dtype,
    )

    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry
            attn_weights = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / scale
            bias_chunk = _chunk_bias_fn(
                q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx
            )
            attn_weights = attn_weights + bias_chunk

            max_score_chunk = torch.maximum(
                prev_max_score_chunk, torch.max(attn_weights, dim=-1)
            )
            max_score_chunk = max_score_chunk.detach()  # Stop gradient
            exp_weights = torch.exp(attn_weights - max_score_chunk[..., None])
            exp_values = torch.einsum("bhqk,bkhd->bqhd", exp_weights, value_chunk)
            correction = rearrange(
                torch.exp(prev_max_score_chunk - max_score_chunk), "b h q -> b q h"
            )[..., None]
            numerator_chunk = numerator_chunk * correction + exp_values
            denominator_chunk = denominator_chunk * torch.exp(
                prev_max_score_chunk - max_score_chunk
            ).sum(dim=-1) + exp_weights.sum(dim=-1)
            return (numerator_chunk, denominator_chunk, max_score_chunk), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            skip_block = torch.tensor(False)
            if causal:
                skip_block = (
                    q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
                )
            if skip_block:
                return carry, None
            else:
                return scan_kv_block(carry, args)

        numerator_chunk, denominator_chunk, max_score_chunk = (
            numerator_chunk.clone(),
            denominator_chunk.clone(),
            max_score_chunk.clone(),
        )
        for k_chunk_idx in range(num_kv):
            key_chunk = k[:, :, k_chunk_idx]
            value_chunk = v[:, :, k_chunk_idx]
            carry = (numerator_chunk, denominator_chunk, max_score_chunk)
            carry, _ = skip_upper_half(carry, (key_chunk, value_chunk, k_chunk_idx))
            numerator_chunk, denominator_chunk, max_score_chunk = carry

        output_chunk = numerator_chunk / rearrange(denominator_chunk, "b h q -> b q h")[
            ..., None
        ].astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)

    # end of scan_attention func
    numerator, denominator, max_score = (
        torch.zeros_like(numerator_chunk),
        torch.zeros_like(denominator_chunk),
        torch.zeros_like(max_score_chunk),
    )
    for q_idx in range(num_q):
        carry = (numerator, denominator, max_score)
        args = (q[q_idx], numerator, denominator, max_score)
        carry = scan_attention(carry, args)
        (numerator, denominator, max_score) = carry

    numerator = torch.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(
        lambda x: rearrange(x, "n b h c -> b h n c"), (denominator, max_score)
    )
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score


def _blockwise_attention_bwd(
    q,
    k,
    v,
    g,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    bias,
    segment_ids,
    causal,
    query_chunk_size,
    key_chunk_size,
    deterministic,
    dropout_rng,
    attn_pdrop,
    dtype,
    policy,
    precision,
    prevent_cse,
):
    batch, q_len, num_heads, dim_per_head = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    batch, kv_len, num_heads, dim_per_head = v.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry

    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(
        lambda x: torch.moveaxis(x, 1, 0), (g, dq, dk, dv, output)
    )

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(
        lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score)
    )

    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    k = k.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    v = v.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    q, k, v = map(lambda x: torch.moveaxis(x, 1, 0), (q, k, v))

    scale = torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = torch.split(dropout_rng, 2)
        attn_dropout = torch.bernoulli(attn_dropout_rng, attn_pdrop).expand(
            batch, num_heads, q_len, kv_len
        )
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size,
        key_chunk_size,
        bias,
        segment_ids,
        deterministic,
        attn_dropout,
        attn_pdrop,
        causal,
        dtype,
    )

    def scan_attention(carry, scan):
        dk, dv = carry
        (
            q_chunk,
            dq_chunk,
            g_chunk,
            output_chunk,
            denominator_chunk,
            max_score_chunk,
            q_chunk_idx,
        ) = scan
        dl_part = torch.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]

        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = torch.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk) / scale
            bias_chunk = _chunk_bias_fn(
                q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx
            )
            attn_weights = attn_weights + bias_chunk
            exp_weights = (
                torch.exp(attn_weights - max_score_chunk[..., None])
                / denominator_chunk[..., None]
            )

            ds = torch.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights
            dq_chunk = dq_chunk + torch.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / scale
            dk_chunk = torch.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / scale
            dv_chunk = torch.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)
            return dq_chunk, (dk_chunk, dv_chunk)

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, k_chunk_idx = args
            skip_block = torch.tensor(False)
            if causal:
                skip_block = (
                    q_chunk_idx_start + q_chunk_idx < k_chunk_idx_start + k_chunk_idx
                )
            if skip_block:
                return carry, (torch.zeros_like(dk), torch.zeros_like(dv))
            else:
                return scan_kv_block(carry, args)

        dq_chunk, (dk_part, dv_part) = skip_upper_half(
            dq_chunk, (k, v, jnp.arange(0, num_kv))
        )
        return (dk + dk_part, dv + dv_part), dq_chunk

    # end of scan_attention func
    dk, dv = dk.clone(), dv.clone()
    dq, dq_chunk = dq.clone(), dq_chunk.clone()

    for q_chunk_idx in range(num_q):
        carry = (dk, dv)
        carry, dq_chunk = scan_attention(
            carry,
            (
                q[:, :, q_chunk_idx],
                dq[:, :, q_chunk_idx],
                g[:, :, q_chunk_idx],
                output[:, :, q_chunk_idx],
                denominator[:, :, q_chunk_idx],
                max_score[:, :, q_chunk_idx],
                q_chunk_idx,
            ),
        )
        dk, dv = carry

    dq = dq.permute(0, 2, 1, 3).reshape(batch, q_len, num_heads, dim_per_head)
    dk = dk.permute(0, 2, 1, 3).reshape(batch, kv_len, num_heads, dim_per_head)
    dv = dv.permute(0, 2, 1, 3).reshape(batch, kv_len, num_heads, dim_per_head)

    return dq, dk, dv


# Blockwise feedforward network for memory-efficient training
def blockwise_ffn(remat_ffn, inputs, chunk_size, deterministic):
    inputs = rearrange(inputs, "b (c n) d -> b c n d", c=chunk_size)

    def scan_ffn(remat_ffn, carry, hidden_states):
        outputs = remat_ffn(hidden_states, deterministic=deterministic)
        return carry, outputs

    scan_axis = inputs.dim() - 2
    output = torch.empty_like(inputs)
    for idx in range(inputs.size(scan_axis)):
        hidden_states = inputs.select(scan_axis, idx)
        _, output_chunk = scan_ffn(remat_ffn, None, hidden_states)
        output.select(scan_axis, idx).copy_(output_chunk)

    output = rearrangerange(output, "b c n d -> b (c n) d")
    return output


def _chunk_attention_bias(
    query_chunk_size,
    key_chunk_size,
    bias,
    segment_ids,
    deterministic,
    attn_dropout,
    attn_pdrop,
    causal,
    dtype,
    query_chunk_idx,
    key_chunk_idx,
):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = torch.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = bias[
            ...,
            key_offset : key_offset + query_chunk_size,
            : min(key_chunk_size, bias.shape[-1]),
        ]

    if segment_ids is not None:
        q_segment_ids = segment_ids[:, query_offset : query_offset + query_chunk_size]
        k_segment_ids = segment_ids[:, key_offset : key_offset + key_chunk_size]
        segment_ids_mask = q_segment_ids[:, :, None] != k_segment_ids[:, None, :]
        segment_ids_mask = segment_ids_mask[:, None]  # B1QK
        segment_ids_bias = segment_ids_mask * torch.finfo(dtype).min
        chunk_bias += segment_ids_bias

    if causal:
        query_idx = torch.arange(query_chunk_size, dtype=torch.int32).unsqueeze(1)
        key_idx = torch.arange(key_chunk_size, dtype=torch.int32).unsqueeze(0)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * torch.finfo(dtype).min
        chunk_bias += causal_mask_value.view(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = attn_dropout[
            ...,
            query_offset : query_offset + query_chunk_size,
            key_offset : key_offset + key_chunk_size,
        ]
        chunk_bias += attn_dropout_slice * torch.finfo(dtype).min
    return chunk_bias.to(dtype)


## Flash attention
def _flash_attention(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        ab,
        segment_ids,
        save_residuals,
        causal,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
    )


def _flash_attention_fwd(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    o, l, m = _flash_attention(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        ab,
        segment_ids,
        True,
        causal,
        sm_scale,
        block_sizes,
        debug,
    )
    return o, l, m


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


## FA Kernel
def _flash_attention_impl(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
    assert block_k_major == block_k, (block_k_major, block_k)
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    acc, l_prev, m_prev = carry
    l_prev, m_prev = map(
        lambda x: x.unsqueeze(-1).expand(*x.shape, MIN_BLOCK_SIZE),
        (l_prev, m_prev),
    )
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    grid = (
        torch.cdiv(batch_size, block_b, rounding_mode="trunc"),
        num_heads,
        torch.cdiv(q_seq_len, block_q, rounding_mode="trunc"),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
    ):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = torch.where(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q,
                    kv_seq_index + k_idx_refx_ref[0],
                    block_k_major,
                ),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(
        batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
    ):
        if causal:
            should_run = below_or_on_diag(
                q_seq_index + q_idx_ref[0],
                block_q,
                kv_seq_index + k_idx_ref[0],
                block_k_major,
            )
            next_kv_index = torch.where(should_run, kv_seq_index, 0)
        else:
            next_kv_index = kv_seq_index

        return (batch_index, 0, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        block_q=block_q,
    )
    out_shape = [q.shape, q.dtype]
    out_specs = [(block_b, 1, block_q, head_dim)]

    if block_k != kv_seq_len:
        # scratch_shape = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
        m_scratch = torch.zeros(
            (block_b, 1, block_q, MIN_BLOCK_SIZE), dtype=torch.float32
        )
        l_scratch = torch.zeros(
            (block_b, 1, block_q, MIN_BLOCK_SIZE), dtype=torch.float32
        )
        acc_scratch = torch.zeros((block_b, 1, block_q, head_dim), dtype=torch.float32)
        out_shape += [m_scratch.shape, l_scratch.shape, acc_scratch.shape]
        out_specs += [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ]
    else:
        assert False
        out_shape += [None, None, None]
        out_specs += [None, None, None]

    if save_residuals:
        out_specs = [
            (block_b, 1, block_q, MIN_BLOCK_SIZE),
            (block_b, 1, block_q, MIN_BLOCK_SIZE),
        ]
        l = torch, zeros(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=toch.float32
        )
        m = torch.zeros(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=torch.float32
        )
        out_shape += (l.shape, m.shape)

    # Assuming ab is a tensor or None
    if ab is not None:
        ab_block_spec = ab_index_map(block_b, block_q, block_k_major)
    else:
        ab_block_spec = None

    if ab is not None:
        ab = ab[:, None].repeat(block_q, axis=1)

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(
            batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref
        ):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(
            batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref
        ):
            del head_index
            if causal:
                next_kv_index = torch.where(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q,
                        kv_seq_index + k_idx_ref[0],
                        block_k_major,
                    ),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        # Assuming q_segment_ids_index_map and kv_segment_ids_index_map are defined PyTorch functions
        q_segment_ids_spec = lambda *args, **kwargs: q_segment_ids_index_map(
            *args, **kwargs, block_b=block_b, block_q=block_q, num_lanes=NUM_LANES
        )
        kv_segment_ids_spec = lambda *args, **kwargs: kv_segment_ids_index_map(
            *args,
            **kwargs,
            block_b=block_b,
            num_sublanes=NUM_SUBLANES,
            block_k_major=block_k_major
        )

        # Assuming segment_ids.q and segment_ids.kv are defined PyTorch tensors
        q_segment_ids = segment_ids.q.expand(batch_size, q_seq_len, NUM_LANES)
        kv_segment_ids = segment_ids.kv.expand(batch_size, NUM_SUBLANES, kv_seq_len)

    in_specs = [
        q_index_map(block_b, 1, block_q, head_dim),
        kv_index_map(block_b, 1, block_k_major, head_dim),
        kv_index_map(block_b, 1, block_k_major, head_dim),
        q_index_map(block_b, 1, block_q, head_dim),
        lm_index_map(block_b, 1, block_q, MIN_BLOCK_SIZE),
        lm_index_map(block_b, 1, block_q, MIN_BLOCK_SIZE),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    # Assuming q_chunk_idx_start, k_chunk_idx_start, q, k, v, acc, l_prev, m_prev, ab, q_segment_ids, kv_segment_ids are defined PyTorch tensors
    o, *aux = kernel(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        acc,
        l_prev,
        m_prev,
        ab,
        q_segment_ids,
        kv_segment_ids,
    )

    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o
