import torch
from torch_geometric.nn.inits import glorot


class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1, in_channels))
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb
