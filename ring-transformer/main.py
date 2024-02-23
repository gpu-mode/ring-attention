import torch
from torch.optim import AdamW
from torch import nn
from einops import rearrange

import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_qkvpacked_func
from flash_attn import flash_attn_qkvpacked_func


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, attn_fn):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.heads = heads
        self.attn_fn = attn_fn

    def forward(self, x):
        # (batch_size, seqlen, hidden_dim)
        x = self.norm(x)

        x = self.qkv_proj(x)
        # (batch_size, seqlen, (3 * nheads * headdim))

        x = rearrange(x, "B S (a nheads headdim) -> B S a nheads headdim", a=3, nheads=self.heads)
        # (batch_size, seqlen, 3, nheads, headdim)

        #x = flash_attn_qkvpacked_func(
        x = self.attn_fn(x, causal=False)

        # (batch_size, seqlen, nheads, headdim)
        x = rearrange(x, "B S nheads headdim -> B S (nheads headdim)")

        # (batch_size, seqlen, hidden_dim)
        x = self.out_proj(x)

        return x


class Layer(nn.Module):
    def __init__(self, hidden_dim, heads, mlp_dim, attn_fn, dropout = 0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, heads, attn_fn)
        self.ffn = FeedForward(hidden_dim, mlp_dim, dropout = dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, num_layers, hidden_dim, heads, mlp_dim, attn_fn):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(Layer(hidden_dim=hidden_dim, heads=heads, mlp_dim=mlp_dim, attn_fn=attn_fn))

    def forward(self, x):
        for l in self.layers:
           x = l(x)
        return x


def broadcast_model(model, source=0):
    for param in model.parameters():
        dist.broadcast(param.data, src=source)


def cross_check():
    B,S,D = 2,4,16

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    dist.broadcast(x, src=0)

    t1 = Transformer(num_layers=2, hidden_dim=D, heads=2, mlp_dim=D*4, attn_fn=flash_attn_qkvpacked_func)
    t1.to(device=device, dtype=dtype)

    broadcast_model(t1, source=0)

    t2 = Transformer(num_layers=2, hidden_dim=D, heads=2, mlp_dim=D*4, attn_fn=ring_flash_attn_qkvpacked_func)
    t2.to(device=device, dtype=dtype)

    for a,b in zip(t1.parameters(), t2.parameters()):
        b.data.copy_(a)

    y1 = t1(x)

    local_x = x.chunk(world_size, dim=1)[rank].detach().clone()

    y2 = t2(local_x)
    print(f"rank={rank}, shape={y2.shape}")
    #print(f"rank {rank}: {y}")

    if rank == 0:
        print("diff: ", y1.chunk(world_size, dim=1)[rank] - y2)


def main():
    B,S,D = 2,4,16

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    y = torch.randn_like(x)
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)

    t = Transformer(num_layers=2, hidden_dim=D, heads=2, mlp_dim=D*4, attn_fn=ring_flash_attn_qkvpacked_func)
    t.to(device=device, dtype=dtype)
    broadcast_model(t, source=0)

    print(f"rank={rank}, weight={t.layers[0].attn.out_proj.weight}")

    local_params = t.parameters()
    optmizer = AdamW(params = local_params, lr=0.001)

    local_x = x.chunk(world_size, dim=1)[rank].detach().clone()
    local_y = y.chunk(world_size, dim=1)[rank].detach().clone()

    loss = nn.MSELoss()

    for i in range(200):

        out = t(local_x)

        l = loss(out, local_y)

        if i % 50 == 0:
            print(f'iter={i}, rank={rank}, loss={l}')

        l.backward()

        for p in local_params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        optmizer.step()

    print(f"rank={rank}, weight={t.layers[0].attn.out_proj.weight}")


if __name__ == '__main__':
    main()
