import torch
import torch.distributed as dist


class BroadComm:
    def __init__(self, process_group: torch.distributed.ProcessGroup):
        self.process_group = process_group

    def recv(self, recv_tensor: torch.Tensor = None) -> torch.Tensor:
        if recv_tensor is None:
            raise ValueError("recv_tensor must be provided")
        # receive from rank 0 which broadcasts the tensor
        if self.process_group is None:
            self.process_group = dist.init_process_group("nccl")
        recv_rank = 0
        dist.recv(recv_tensor, src=recv_rank, group=self.process_group)
        return recv_tensor

    def broadcast(self, send_tensor: torch.Tensor, src: int) -> None:
        if self.process_group is not None:
            src = dist.get_rank(self.process_group)
        dist.broadcast(send_tensor, src=src, group=self.process_group)


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    dtype = torch.bfloat16

    # Initialize tensor on all ranks, but it will be overwritten by the broadcast
    dout = torch.empty(3, device=device, dtype=dtype)

    # Rank 0 has the tensor to be broadcasted
    if rank == 0:
        dout = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)

    # Broadcast dout tensor from rank 0 to all ranks
    dist.broadcast(dout, src=0)

    dist.barrier()

    print(f"rank {rank} dout: {dout}")
