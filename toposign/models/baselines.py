"""
Thin wrappers over the existing signed GNN baseline models.

Each wrapper normalizes the forward interface to:
    z, log_prob, pred, prob = model(data, pi=None)

Baselines for Table 2:
    SSSNET   - torch_geometric_signed_directed.nn.SSSNET_node_clustering
    MSGNN    - MSGNN/MSGNN.py  MSGNN_node_classification
    SigMaNet - SigMaNet/Signum.py
    DSGC     - DSGC/encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric_signed_directed.nn import SSSNET_node_clustering
from toposign.models.backbones.msgnn.MSGNN import MSGNN_node_classification
from toposign.models.backbones.sigmanet.Signum import SigMaNet_node_prediction_one_laplacian
from toposign.models.backbones.sigmanet import laplacian as sigmanet_laplacian
from toposign.models.backbones.dsgc.encoder import DSGC as _DSGC


class SSSNETWrapper(nn.Module):
    """SSSNET node clustering baseline."""

    def __init__(self, num_features: int, hidden: int, label_dim: int,
                 hop: int = 2, tau: float = 0.5, dropout: float = 0.5):
        super().__init__()
        self.embed_dim = 4 * hidden
        self.model = SSSNET_node_clustering(
            nfeat=num_features,
            hidden=hidden,
            nclass=label_dim,
            dropout=dropout,
            hop=hop,
            fill_value=tau,
            directed=True,
        )

    def forward(self, data, pi=None):
        z, log_prob, pred, prob = self.model(
            data.edge_index_p, data.edge_weight_p,
            data.edge_index_n, data.edge_weight_n,
            data.x,
        )
        return z, log_prob, pred, prob


class MSGNNWrapper(nn.Module):
    """MSGNN node classification baseline."""

    def __init__(self, num_features: int, hidden: int, label_dim: int,
                 K: int = 1, q: float = 0.25, num_layers: int = 2,
                 dropout: float = 0.5, normalization: str = 'sym'):
        super().__init__()
        self.embed_dim = 2 * hidden
        self.model = MSGNN_node_classification(
            num_features=num_features,
            hidden=hidden,
            label_dim=label_dim,
            K=K, q=q, layer=num_layers,
            dropout=dropout,
            normalization=normalization,
        )

    def forward(self, data, pi=None):
        z, log_prob, pred, prob = self.model(
            data.x, data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
        )
        return z, log_prob, pred, prob


class SigMaNetWrapper(nn.Module):
    """SigMaNet node prediction baseline (lazy-initializes Laplacian)."""

    def __init__(self, num_features: int, label_dim: int,
                 hidden: int = 32,
                 K: int = 1, dropout: float = 0.5,
                 normalization: str = 'sym'):
        super().__init__()
        self.embed_dim = 2 * hidden
        self._num_features  = num_features
        self._label_dim     = label_dim
        self._hidden        = hidden
        self._K             = K
        self._dropout       = dropout
        self._normalization = normalization
        self.model: Optional[nn.Module] = None

    def set_laplacian(self, edge_index, edge_weight, num_nodes, device):
        lap_edge_index, norm_real, norm_imag = sigmanet_laplacian.process_magnetic_laplacian(
            edge_index=edge_index,
            gcn=False,
            net_flow=False,
            x_real=torch.ones(num_nodes, 1, device=device),
            edge_weight=edge_weight,
            normalization='sym',
            return_lambda_max=False,
        )
        lap_edge_index = lap_edge_index.to(device)
        norm_real = norm_real.to(device)
        norm_imag = norm_imag.to(device)
        self.model = SigMaNet_node_prediction_one_laplacian(
            K=self._K,
            num_features=self._num_features,
            hidden=self._hidden,
            label_dim=self._label_dim,
            dropout=self._dropout,
            normalization=self._normalization,
            unwind=True,
            edge_index=lap_edge_index,
            norm_real=norm_real,
            norm_imag=norm_imag,
        ).to(device)

    def parameters(self, recurse: bool = True):
        if self.model is not None:
            return self.model.parameters(recurse)
        return iter([])

    def forward(self, data, pi=None):
        assert self.model is not None, "Call set_laplacian() before forward()."
        out = self.model(data.x, data.x)
        if isinstance(out, tuple):
            return out
        log_prob = out
        pred = log_prob.argmax(dim=1)
        prob = log_prob.exp()
        return log_prob, log_prob, pred, prob


class DSGCWrapper(nn.Module):
    """DSGC (Deep Signed Graph Clustering) baseline."""

    def __init__(self, num_features: int, hidden: int, label_dim: int,
                 hop: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embed_dim = 4 * hidden
        self.model = _DSGC(
            nfeat=num_features,
            hidden=hidden,
            nclass=label_dim,
            dropout=dropout,
            hop=hop,
            directed=True,
        )

    def forward(self, data, pi=None):
        num_nodes = data.x.size(0)
        device = data.x.device
        A_p = _dense_adjacency(
            data.edge_index_p, data.edge_weight_p, num_nodes, device, data.x.dtype
        )
        A_n = _dense_adjacency(
            data.edge_index_n, data.edge_weight_n, num_nodes, device, data.x.dtype
        )
        _, _, _, z, pred, prob = self.model(A_p, A_n, data.x, A_p.t(), A_n.t())
        log_prob = torch.log(prob + 1e-8)
        return z, log_prob, pred, prob


def build_baseline(method: str, num_features: int, hidden: int,
                   label_dim: int, args) -> nn.Module:
    """Instantiate a baseline model by name."""
    if method == 'SSSNET':
        return SSSNETWrapper(num_features, hidden, label_dim,
                             hop=getattr(args, 'hop', 2),
                             tau=getattr(args, 'tau', 0.5),
                             dropout=args.dropout)
    elif method == 'MSGNN':
        return MSGNNWrapper(num_features, hidden, label_dim,
                            K=args.K, q=args.q,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            normalization=args.normalization)
    elif method == 'SigMaNet':
        sigmanet_hidden = getattr(args, 'sigmanet_hidden', 1)
        return SigMaNetWrapper(num_features, label_dim,
                               hidden=sigmanet_hidden,
                               K=args.K,
                               dropout=args.dropout,
                               normalization=args.normalization)
    elif method == 'DSGC':
        return DSGCWrapper(num_features, hidden, label_dim,
                           hop=getattr(args, 'hop', 2),
                           dropout=args.dropout)
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def _dense_adjacency(edge_index, edge_weight, num_nodes, device, dtype):
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    edge_index = edge_index.to(device)
    if edge_weight is None:
        values = torch.ones(edge_index.size(1), dtype=dtype, device=device)
    else:
        values = edge_weight.to(device=device, dtype=dtype).abs()
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    adj[edge_index[0], edge_index[1]] = values
    return adj
