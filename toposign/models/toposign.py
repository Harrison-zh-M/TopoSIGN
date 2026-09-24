"""
TopoSIGN: Topology-guided pre-training and prompt learning for signed graphs.

Architecture (Figure 1 of the paper):
  Z_G  = SGE(x, edge_index, edge_weight)      structural branch (2*hidden)
  Z_T  = TPL(pi_tensor)                         topological branch (topo_hidden)
  Z    = [Z_G || Z_T]  in R^{N x (2*hidden + topo_hidden)}  fused embedding

  Pre-training:  edge MLP on Z -> BCE loss for link sign prediction
  Fine-tuning:   cluster prompts on Z -> NLL loss on seed nodes

Ablation variants (Table 3):
  tda_type='none'     -> SGE only (no Topo)
  tda_type='positive' -> SGE + positive-edge Topo
  tda_type='signed'   -> SGE + signed degree-vector Topo  (ours)
  structural=False    -> Topo only (no SGE)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from toposign.models.backbones.msgnn.MSConv import MSConv


class TPL(nn.Module):
    """Topological Projection Layer: single Linear → BN → ReLU → Dropout."""

    def __init__(self, pi_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(pi_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, pi: torch.Tensor) -> torch.Tensor:
        if pi.dim() > 2:
            pi = pi.view(pi.size(0), -1)
        return self.proj(pi)


class TopoSIGN(nn.Module):
    """
    TopoSIGN node-level model combining SGE and TPL branches.

    Args:
        num_features (int): Input node-feature dimension.
        pi_dim (int):       Flattened persistence-image size (0 if no TDA).
        hidden (int):       Hidden channels for SGE layers.
        topo_hidden (int):  Output dimension of the TPL projection.
        label_dim (int):    Number of output cluster classes.
        num_layers (int):   Number of MSConv layers.
        q (float):          Magnetic-Laplacian charge parameter.
        K (int):            Chebyshev polynomial order.
        dropout (float):    Dropout rate.
        normalization (str): Laplacian normalization ('sym' or None).
        tda_type (str):     'signed' (ours) | 'positive' | 'none'.
        use_structural (bool): If False, disable SGE (Topo-only ablation).
    """

    def __init__(
        self,
        num_features: int,
        pi_dim: int,
        hidden: int = 16,
        topo_hidden: int = 16,
        label_dim: int = 2,
        num_layers: int = 2,
        q: float = 0.25,
        K: int = 1,
        dropout: float = 0.5,
        normalization: str = 'sym',
        tda_type: str = 'signed',
        use_structural: bool = True,
    ):
        super().__init__()

        self.tda_type       = tda_type
        self.use_structural = use_structural
        self.use_tda        = (tda_type != 'none') and (pi_dim > 0)
        self.dropout_val    = dropout

        if use_structural:
            chebs = nn.ModuleList()
            chebs.append(MSConv(
                in_channels=num_features, out_channels=hidden,
                K=K, q=q, trainable_q=False, normalization=normalization,
            ))
            for _ in range(1, num_layers):
                chebs.append(MSConv(
                    in_channels=hidden, out_channels=hidden,
                    K=K, q=q, trainable_q=False, normalization=normalization,
                ))
            self.chebs = chebs
            struct_dim = 2 * hidden
        else:
            self.chebs = None
            struct_dim = 0

        if self.use_tda:
            self.tpl     = TPL(pi_dim, topo_hidden, dropout)
            topo_dim     = topo_hidden
        else:
            self.tpl  = None
            topo_dim  = 0

        fused_dim = struct_dim + topo_dim
        assert fused_dim > 0, "At least one of structural or TDA branch must be enabled."
        self.classifier = nn.Conv1d(fused_dim, label_dim, kernel_size=1)

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        pi: Optional[torch.Tensor] = None,
    ):
        parts = []

        if self.use_structural and self.chebs is not None:
            for cheb in self.chebs:
                real, imag = cheb(real, imag, edge_index, edge_weight)
            parts.append(torch.cat((real, imag), dim=-1))

        if self.use_tda and pi is not None and self.tpl is not None:
            parts.append(self.tpl(pi))

        z = torch.cat(parts, dim=-1)

        if self.dropout_val > 0:
            z = F.dropout(z, self.dropout_val, training=self.training)

        x = z.unsqueeze(0).permute(0, 2, 1)
        logits = self.classifier(x)
        output = torch.transpose(
            F.log_softmax(logits, dim=1)[0], 0, 1
        )

        predictions_cluster = torch.argmax(output, dim=1)
        prob = F.softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob

    def encode(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        pi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z, _, _, _ = self.forward(real, imag, edge_index, edge_weight, pi)
        return z


class EdgeSignDecoder(nn.Module):
    """
    MLP decoder that scores a pair (z_u, z_v) as positive or negative link.
    Input:  concatenated embeddings [z_u || z_v], shape [E, 2*embed_dim].
    Output: logit per edge, shape [E].  BCE loss is applied outside.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)
