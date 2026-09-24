import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class _SimpleMeanConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j


class GPPTPrompt(torch.nn.Module):
    def __init__(self, n_hidden, center_num, n_classes, device):
        super().__init__()
        self.center_num = center_num
        self.n_classes = n_classes
        self.device = device
        self.StructureToken = torch.nn.Linear(n_hidden, center_num, bias=False).to(device)
        self.TaskToken = torch.nn.ModuleList(
            [torch.nn.Linear(2 * n_hidden, n_classes, bias=False) for _ in range(center_num)]
        ).to(device)

    def weigth_init(self, h, edge_index, label, index):
        conv = _SimpleMeanConv()
        h = conv(h, edge_index)
        features = h[index]
        labels = label[index.long()]

        k = min(self.center_num, features.shape[0])
        cluster = KMeans(n_clusters=k, random_state=0).fit(features.detach().cpu())
        self.StructureToken.weight.data = torch.FloatTensor(
            cluster.cluster_centers_
        ).to(self.device).clone().detach()

        p = [features[labels == i].mean(dim=0).view(1, -1) for i in range(self.n_classes)]
        temp = torch.cat(p, dim=0).to(self.device)
        for i in range(self.center_num):
            self.TaskToken[i].weight.data = temp.clone().detach()

    def get_TaskToken(self):
        return [param for name, param in self.named_parameters()
                if name.startswith('TaskToken.')]

    def get_StructureToken(self):
        for name, param in self.named_parameters():
            if name.startswith('StructureToken.weight'):
                return param

    def forward(self, h, edge_index):
        conv = _SimpleMeanConv()
        h = conv(h, edge_index)
        self.fea = h
        out = self.StructureToken(h)
        index = torch.argmax(out, dim=1)
        result = torch.zeros(h.shape[0], self.n_classes).to(h.device)
        for i in range(self.center_num):
            mask = index == i
            if mask.any():
                result[mask] = self.TaskToken[i](h[mask])
        return result
