import pytest
import torch
from flash_attn import flash_attn_varlen_func
from flash_attn_varlen_triton import flash_attn_varlen_triton

@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_varlen_compare_tri_dao(num_heads: int, headdim: int, dtype: torch.dtype, num_seq: int = 10):
    device = torch.device("cuda", 0)

    seqlens = []

    with device:
        seqlens = torch.randint(1, 10, (num_seq,), dtype=torch.int32)
        cu_seqlens = torch.cat((torch.tensor([0], dtype=torch.int32), seqlens.cumsum(dim=0, dtype=torch.int32)))
        total_tokens = cu_seqlens[-1]
        max_seqlen = seqlens.max()

        q = torch.randn(total_tokens, num_heads, headdim, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        a = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_k=max_seqlen, max_seqlen_q=max_seqlen)
        b = flash_attn_varlen_triton(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)

        #assert torch.allclose(a, b)
        abs_delta = torch.abs(a-b).sum()

        print('abs_delta', abs_delta.item())
