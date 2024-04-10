import itertools
import math

import torch


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def flash_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
):
    # N.B.: This uses the PyTorch SDPA tensor shape of batch, head_no, seq_len, head_dim

    # this is a verbatim adaptation of the FlashAttention 2 pseudocode
    # (with a bit of ignorance to memory locations)

    # TODO: handle inputs that are not a multiple of B_r / B_c

    *batch, N_inp, d = K.shape
    *_, N_out, _ = Q.shape

    # assert shape compat

    O = V.new_zeros(*batch, N_out, d)
    L = V.new_zeros(*batch, N_out, 1)

    dtype = O.dtype
    device = O.device

    neginf = torch.tensor(-math.inf, dtype=Q.dtype, device=Q.device)

    B_c = 16
    B_r = 16
    T_c = (N_inp + B_c - 1) // B_c
    T_r = (N_out + B_r - 1) // B_r

    if scale is None:
        scale = 1 / math.sqrt(d)

    for block in itertools.product(*(range(s) for s in batch)):
        # Q and O L split into T_r; K, V in T_c blocks
        for i in range(T_r):
            Q_i = Q[block][i * B_r : (i + 1) * B_r]
            O_i = torch.zeros(B_r, d, device=device, dtype=dtype)
            l_i = torch.zeros(B_r, 1, device=device, dtype=dtype)
            m_i = torch.full((B_r, 1), -math.inf, device=device, dtype=dtype)
            last_m_i = m_i
            for j in range(T_c):
                if is_causal and j * B_c > (i + 1) * B_r - 1:
                    break

                K_j = K[block][j * B_c : (j + 1) * B_c]
                V_j = V[block][j * B_c : (j + 1) * B_c]
                S_i = scale * (Q_i @ K_j.T)
                if is_causal and i * B_r < (j + 1) * B_c - 1:
                    mask = (
                        torch.arange(
                            i * B_r, (i + 1) * B_r, device=device, dtype=dtype
                        )[:, None]
                        >= torch.arange(
                            j * B_c, (j + 1) * B_c, device=device, dtype=dtype
                        )[None, :]
                    )
                    S_i = torch.where(mask, S_i, neginf)

                m_i = torch.maximum(m_i, S_i.max(dim=-1, keepdim=True).values)
                P_i = torch.exp(S_i - m_i)
                l_i = torch.exp(last_m_i - m_i) * l_i + P_i.sum(dim=-1, keepdim=True)
                O_i = torch.exp(last_m_i - m_i) * O_i + P_i @ V_j
                last_m_i = m_i
            O_i = (1.0 / l_i) * O_i
            L_i = m_i + torch.log(l_i)
            O[block][i * B_r : (i + 1) * B_r] = O_i
            L[block][i * B_r : (i + 1) * B_r] = L_i
    return O, L


def test_flash_attention_reference(
    seqlen: int = 32,
    num_heads: int = 4,
    headdim: int = 64,
    dtype=torch.float32,
    device: str = "cuda:0",
):
    device = torch.device(device)
    with device:
        q = torch.randn(seqlen, num_heads, headdim, dtype=dtype).transpose(-2, -3)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

    for is_causal in [False, True]:
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )
        actual, _ = flash_attention_reference(q, k, v, is_causal=is_causal)
        torch.testing.assert_close(actual, expected, msg=f"with {is_causal=}")


if __name__ == "__main__":
    test_flash_attention_reference()
