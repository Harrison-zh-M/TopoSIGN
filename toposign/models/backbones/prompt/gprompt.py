import torch


class Gprompt(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, node_embeddings):
        return node_embeddings * self.weight
