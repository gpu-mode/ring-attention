import torch
import unittest
from ring_attn.ring_attention import RingAttention, RingAttentionStandard


class TestRingAttention(unittest.TestCase):
    def test_ring_attention(self):
        # Define the parameters for the RingAttention class
        dim = 512
        dim_head = 64
        heads = 8
        causal = True
        auto_shard_seq = True
        ring_attn = True
        ring_seq_size = 512

        # Create a RingAttention instance
        ring_attention = RingAttention(
            dim, dim_head, heads, causal, auto_shard_seq, ring_attn, ring_seq_size
        )

        # Generate some input tensors
        q = torch.randn(1, 1024, heads, dim_head)
        k = torch.randn(1, 1024, heads, dim_head)
        v = torch.randn(1, 1024, heads, dim_head)
        attn_bias = torch.zeros(1, 1024, 1024)
        segment_ids = torch.zeros(1, 1024)
        axis_name = None  # Assuming axis_name is not used in the forward method
        float32_logits = (
            False  # Assuming float32_logits is not used in the forward method
        )
        blockwise_kwargs = (
            {}
        )  # Assuming blockwise_kwargs is not used in the forward method

        # Call the apply method
        output = ring_attention.apply(
            q, k, v, attn_bias, segment_ids, axis_name, float32_logits, blockwise_kwargs
        )

        # Check the output shape
        self.assertEqual(output.shape, q.shape)

        # Optionally, check the output values
        # For example, you might check that the output is not all zeros
        self.assertFalse(torch.all(output == 0))


if __name__ == "__main__":
    unittest.main()
