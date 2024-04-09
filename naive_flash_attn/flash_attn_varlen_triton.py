from typing import Optional
import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["max_seqlen_k"] % args["BLOCK_N"] == 0,
        "GQA": lambda args: args["num_kv_groups"] > 1,
    }
)
@triton.jit
def _fwd_kernel_varlen(
    Q,
    K,
    V,
    Out,
    softmax_scale,
    stride_qh,  # Q : [h, m, d]
    stride_qm,
    stride_kh,  # K : [h, n, d]
    stride_kn,
    stride_vh,  # V : [h, n, d]
    stride_vn,
    stride_oh,  # output: [h, m, d]
    stride_om,
    num_q_heads: int,
    num_kv_groups: int,
    cu_seqlen_q,
    cu_seqlen_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    headdim: int,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA: tl.constexpr,
):
    ## variable length sequences
    ## (max_seqlen_q / BLOCK_M, batch_size * numheads)

    start_m = tl.program_id(0)  # q block
    off_hb = tl.program_id(1)
    off_b = off_hb // num_q_heads  # batch index
    off_h = off_hb % num_q_heads  # head index

    begin_q = tl.load(cu_seqlen_q + off_b)
    end_q = tl.load(cu_seqlen_q + off_b + 1)
    begin_k = tl.load(cu_seqlen_k + off_b)
    end_k = tl.load(cu_seqlen_k + off_b + 1)

    offs_m = start_m * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # q row block offsets, BLOCK_M is block over rows of queries
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_n = tl.arange(0, BLOCK_N)  # BLOCK_N block over rows of keys

    q_ptrs = (
        Q
        + off_h * stride_qh
        + ((offs_m[:, None] + begin_q) * stride_qm + offs_d[None, :])
    )  # q pointers: [BLOCK_M, BLOCK_HEADDIM]
    if GQA:
        off_kv_h = off_h // num_kv_groups
        k_ptrs = (
            K
            + off_kv_h * stride_kh
            + ((offs_n[:, None] + begin_k) * stride_kn + offs_d[None, :])
        )  # k pointers: [BLOCK_N, BLOCK_HEADDIM]
        v_ptrs = (
            V
            + off_kv_h * stride_vh
            + ((offs_n[:, None] + begin_k) * stride_vn + offs_d[None, :])
        )  # v pointers: [BLOCK_N, BLOCK_HEADDIM]
    else:
        k_ptrs = (
            K
            + off_h * stride_kh
            + ((offs_n[:, None] + begin_k) * stride_kn + offs_d[None, :])
        )  # k pointers: [BLOCK_N, BLOCK_HEADDIM]
        v_ptrs = (
            V
            + off_h * stride_vh
            + ((offs_n[:, None] + begin_k) * stride_vn + offs_d[None, :])
        )  # v pointers: [BLOCK_N, BLOCK_HEADDIM]

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # log sum exp
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # current maximum
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)  # accumulated output

    seqlen_q = tl.minimum(end_q - begin_q, max_seqlen_q)
    seqlen_k = tl.minimum(end_k - begin_k, max_seqlen_k)

    # load q: it will stay in SRAM throughout
    q = tl.load(
        q_ptrs,
        mask=((offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)),
        other=0.0,
    )  # query elements: [BLOCK_M, BLOCK_HEADDIM]

    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )  # key elements: [BLOCK_N, BLOCK_HEADDIM]

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Need to mask out otherwise the softmax is wrong
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(
                offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
            )

        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # -- update output accumulator --
        acc_o = acc_o * acc_o_scale[:, None]

        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )  # value elements: [BLOCK_N, BLOCK_HEADDIM]

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics --
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_h * stride_oh
        + ((offs_m[:, None] + begin_q) * stride_om + offs_d[None, :])
    )

    tl.store(
        out_ptrs,
        acc_o,
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
    )


def flash_attn_varlen_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    q: (total_tokens_q, nheads, headdim)
    k: (total_tokens_k, nheads, headdim)
    v: (total_tokens_k, nheads, headdim)
    cu_seqlens_q: (batch_size + 1)
    cu_seqlens_k: (batch_size + 1)
    max_seqlen_q: (int) maximum query sequence length
    max_seqlen_k: (int) maximum keys and values sequence length
    """

    # shape constraints
    total_tokens_q, num_q_heads, d = q.shape
    num_kv_heads = k.size(-2)
    total_tokens_k = k.shape[0]

    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    assert max_seqlen_q >= 0 and max_seqlen_k >= 0

    assert (
        num_q_heads % num_kv_heads == 0
    ), "number of query heads must be divisible by the number of key heads"
    num_kv_groups = num_q_heads // num_kv_heads

    assert k.shape == (total_tokens_k, num_kv_heads, d)
    assert v.shape == (total_tokens_k, num_kv_heads, d)
    assert d <= 128, "FlashAttention only supports head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or d**-0.5
    assert cu_seqlens_q[-1] <= total_tokens_q
    assert cu_seqlens_k[-1] <= total_tokens_k

    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8

    batch = cu_seqlens_q.size(0) - 1

    # 2D launch: (cdiv(max_seqlen_q, BLOCK_M), batch * nheads)
    grid = lambda META: (
        triton.cdiv(max_seqlen_q, META["BLOCK_M"]),
        batch * num_q_heads,
    )
    _fwd_kernel_varlen[grid](
        q,
        k,
        v,
        o,
        softmax_scale,
        q.stride(1),  # Q : [h, m, d]
        q.stride(0),
        k.stride(1),  # K : [h, n, d]
        k.stride(0),
        v.stride(1),  # V : [h, n, d]
        v.stride(0),
        o.stride(1),  # output: [h, m, d]
        o.stride(0),
        num_q_heads,
        num_kv_groups,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        d,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
