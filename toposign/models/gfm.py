"""
Signed-graph adaptations of GFM prompt-learning methods.

Methods covered:
    SignedGPPT      - GPPT prompt tokens   + signed backbone (Table 1, 5, 6)
    SignedGprompt   - Gprompt soft prompt  + signed backbone (Table 1, 5, 6)
    SignedGPF       - GPF feature prompt   + signed backbone (Table 1, 5, 6)

All expose:
    encode(data, pi=None)           -> node embeddings [N, embed_dim]
    forward(data, pi=None)          -> (z, log_prob, pred, prob)
"""

import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from toposign.models.backbones.prompt.gppt import GPPTPrompt
from toposign.models.backbones.prompt.gpf import GPF
from toposign.models.backbones.prompt.gprompt import Gprompt

from toposign.models.toposign import TopoSIGN, TPL, EdgeSignDecoder


def _prepare_graph_dependent_encoder(encoder: nn.Module, data, device) -> None:
    """Initialize encoders whose trainable parameters depend on a graph."""
    if not hasattr(encoder, 'set_laplacian'):
        return
    if getattr(encoder, 'model', None) is not None:
        return
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
    encoder.set_laplacian(edge_index, edge_weight, data.x.size(0), device)


class MagNetTopoEncoder(nn.Module):
    """MagNet structural encoder with optional positive-subgraph Topo branch."""

    def __init__(self, num_features: int, pi_dim: int, hidden: int,
                 topo_hidden: int, label_dim: int, num_layers: int,
                 K: int, q: float, dropout: float, normalization: str,
                 tda_type: str = 'positive'):
        super().__init__()
        _magnet_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'magnet'))
        if _magnet_src not in sys.path:
            sys.path.insert(0, _magnet_src)
        from MagNetConv import MagNetConv
        from complex_relu import complex_relu_layer

        self.chebs = nn.ModuleList()
        self.chebs.append(MagNetConv(
            in_channels=num_features, out_channels=hidden,
            K=K, q=q, trainable_q=False, normalization=normalization,
        ))
        for _ in range(1, num_layers):
            self.chebs.append(MagNetConv(
                in_channels=hidden, out_channels=hidden,
                K=K, q=q, trainable_q=False, normalization=normalization,
            ))
        self.complex_relu = complex_relu_layer()
        self.dropout_val = dropout
        self.use_tda = tda_type != 'none' and pi_dim > 0
        self.tpl = TPL(pi_dim, topo_hidden, dropout) if self.use_tda else None
        fused_dim = 2 * hidden + (topo_hidden if self.use_tda else 0)
        self.classifier = nn.Conv1d(fused_dim, label_dim, kernel_size=1)

    def forward(self, real, imag, edge_index, edge_weight=None, pi=None):
        edge_weight_mag = edge_weight.abs() if edge_weight is not None else None
        for cheb in self.chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight_mag)
            real, imag = self.complex_relu(real, imag)

        parts = [torch.cat((real, imag), dim=-1)]
        if self.use_tda and pi is not None and self.tpl is not None:
            parts.append(self.tpl(pi))
        z = torch.cat(parts, dim=-1)
        if self.dropout_val > 0:
            z = F.dropout(z, self.dropout_val, training=self.training)

        x = z.unsqueeze(0).permute(0, 2, 1)
        logits = self.classifier(x)
        output = torch.transpose(F.log_softmax(logits, dim=1)[0], 0, 1)
        pred = output.argmax(dim=1)
        prob = F.softmax(output, dim=1)
        return F.normalize(z), output, pred, prob

    def encode(self, real, imag, edge_index, edge_weight=None, pi=None):
        z, _, _, _ = self.forward(real, imag, edge_index, edge_weight, pi)
        return z


class _BaselineTopoEncoder(nn.Module):
    """Two-branch encoder: any signed GNN baseline (structural) + TPL (topological)."""

    def __init__(self, backbone: str, num_features: int, pi_dim: int,
                 hidden: int, topo_hidden: int, args):
        super().__init__()
        from toposign.models.baselines import build_baseline
        self._backbone = build_baseline(backbone, num_features, hidden, 2, args)
        backbone_dim = getattr(self._backbone, 'embed_dim', 2 * hidden)
        self._tpl = TPL(pi_dim, topo_hidden, dropout=args.dropout)
        self.embed_dim = backbone_dim + topo_hidden
        if hasattr(self._backbone, 'set_laplacian'):
            self.set_laplacian = self._backbone.set_laplacian

    def forward(self, data, pi=None):
        z_g, _, _, _ = self._backbone(data, None)
        if pi is not None:
            z_t = self._tpl(pi.view(z_g.size(0), -1).to(z_g.device))
            z = torch.cat([z_g, z_t], dim=-1)
        else:
            z = z_g
        return z, None, None, None


def _make_signed_encoder(backbone: str, num_features: int, pi_dim: int,
                          hidden: int, tda_type: str, args):
    """Build the signed GNN encoder backbone."""
    if backbone == 'TopoDIG':
        topo_hidden = getattr(args, 'topo_hidden', hidden)
        model = MagNetTopoEncoder(
            num_features=num_features,
            pi_dim=pi_dim,
            hidden=hidden,
            topo_hidden=topo_hidden,
            label_dim=2,
            num_layers=args.num_layers,
            K=args.K,
            q=args.q,
            dropout=args.dropout,
            normalization=args.normalization,
            tda_type=tda_type,
        )
        embed_dim = 2 * hidden + (topo_hidden if tda_type != 'none' and pi_dim > 0 else 0)
        return model, embed_dim

    if backbone in ('TopoSIGN', 'MSGNN'):
        topo_hidden = getattr(args, 'topo_hidden', hidden)
        use_tda = (backbone == 'TopoSIGN')
        use_structural = (
            bool(getattr(args, 'use_structural', True))
            if backbone == 'TopoSIGN'
            else True
        )
        model = TopoSIGN(
            num_features=num_features,
            pi_dim=pi_dim,
            hidden=hidden,
            topo_hidden=topo_hidden,
            label_dim=2,
            num_layers=args.num_layers,
            K=args.K, q=args.q,
            dropout=args.dropout,
            normalization=args.normalization,
            tda_type=tda_type if use_tda else 'none',
            use_structural=use_structural,
        )
        if use_tda:
            embed_dim = (
                (2 * hidden if use_structural else 0)
                + (topo_hidden if tda_type != 'none' and pi_dim > 0 else 0)
            )
        else:
            embed_dim = 2 * hidden
    else:
        from toposign.models.baselines import build_baseline
        if tda_type != 'none' and pi_dim > 0:
            topo_hidden = getattr(args, 'topo_hidden', hidden)
            model = _BaselineTopoEncoder(backbone, num_features, pi_dim, hidden, topo_hidden, args)
        else:
            model = build_baseline(backbone, num_features, hidden, 2, args)
        embed_dim = getattr(model, 'embed_dim', 2 * hidden)
    return model, embed_dim


class SignedGPPT(nn.Module):
    """GPPT with a signed GNN encoder backbone."""

    def __init__(self, backbone: str, num_features: int, pi_dim: int,
                 hidden: int, label_dim: int, center_num: int,
                 tda_type: str, args, device):
        super().__init__()
        self.device = device
        self.encoder, self.embed_dim = _make_signed_encoder(
            backbone, num_features, pi_dim, hidden, tda_type, args
        )
        self.prompt = GPPTPrompt(
            n_hidden=self.embed_dim,
            center_num=center_num,
            n_classes=label_dim,
            device=device,
        )
        self.edge_decoder = EdgeSignDecoder(self.embed_dim)

    def prepare_graph(self, data):
        _prepare_graph_dependent_encoder(self.encoder, data, self.device)

    def encode(self, data, pi=None):
        self.prepare_graph(data)
        if hasattr(self.encoder, 'encode'):
            return self.encoder.encode(data.x, data.x, data.edge_index, data.edge_weight, pi)
        z, _, _, _ = self.encoder(data, pi)
        return z

    def pretrain_step(self, data, pi, train_edges, train_labels, optimizer):
        self.encoder.train()
        self.edge_decoder.train()
        optimizer.zero_grad()
        z = self.encode(data, pi)
        logits = self.edge_decoder(z, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    def init_prompt(self, data, pi, seed_mask):
        self.encoder.eval()
        with torch.no_grad():
            z = self.encode(data, pi)
        self.prompt.weigth_init(z, data.edge_index, data.y, seed_mask.nonzero().squeeze())

    def forward(self, data, pi=None):
        z = self.encode(data, pi)
        out = self.prompt(z, data.edge_index)
        log_prob = F.log_softmax(out, dim=1)
        pred     = out.argmax(dim=1)
        prob     = F.softmax(out, dim=1)
        return z, log_prob, pred, prob


class SignedGprompt(nn.Module):
    """Gprompt with a signed GNN encoder backbone."""

    def __init__(self, backbone: str, num_features: int, pi_dim: int,
                 hidden: int, label_dim: int,
                 tda_type: str, args, device):
        super().__init__()
        self.device = device
        self.encoder, self.embed_dim = _make_signed_encoder(
            backbone, num_features, pi_dim, hidden, tda_type, args
        )
        self.prompt = Gprompt(input_dim=self.embed_dim)
        self.answering = nn.Sequential(
            nn.Linear(self.embed_dim, label_dim),
            nn.Softmax(dim=1),
        )
        self.edge_decoder = EdgeSignDecoder(self.embed_dim)

    def prepare_graph(self, data):
        _prepare_graph_dependent_encoder(self.encoder, data, self.device)

    def encode(self, data, pi=None):
        self.prepare_graph(data)
        if hasattr(self.encoder, 'encode'):
            return self.encoder.encode(data.x, data.x, data.edge_index, data.edge_weight, pi)
        z, _, _, _ = self.encoder(data, pi)
        return z

    def pretrain_step(self, data, pi, train_edges, train_labels, optimizer):
        self.encoder.train()
        self.edge_decoder.train()
        optimizer.zero_grad()
        z = self.encode(data, pi)
        logits = self.edge_decoder(z, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    def forward(self, data, pi=None):
        z = self.encode(data, pi)
        z_prompted = self.prompt(z)
        out = self.answering(z_prompted)
        log_prob = torch.log(out + 1e-8)
        pred     = out.argmax(dim=1)
        return z, log_prob, pred, out


class SignedGPF(nn.Module):
    """GPF (Graph Prompt Feature) with a signed GNN encoder backbone."""

    def __init__(self, backbone: str, num_features: int, pi_dim: int,
                 hidden: int, label_dim: int,
                 tda_type: str, args, device):
        super().__init__()
        self.device = device
        self.prompt = GPF(in_channels=num_features)
        self.encoder, self.embed_dim = _make_signed_encoder(
            backbone, num_features, pi_dim, hidden, tda_type, args
        )
        self.answering = nn.Linear(self.embed_dim, label_dim)
        self.edge_decoder = EdgeSignDecoder(self.embed_dim)

    def prepare_graph(self, data):
        _prepare_graph_dependent_encoder(self.encoder, data, self.device)

    def _encode_with_gpf(self, data, pi=None):
        self.prepare_graph(data)
        prompted_x = self.prompt.add(data.x)
        if hasattr(self.encoder, 'encode'):
            return self.encoder.encode(
                prompted_x, prompted_x,
                data.edge_index, data.edge_weight, pi
            )
        data_prompted = data.clone() if hasattr(data, 'clone') else type('DataObj', (), {})()
        if not hasattr(data, 'clone'):
            data_prompted.edge_index = data.edge_index
            data_prompted.edge_weight = data.edge_weight
            data_prompted.y = getattr(data, 'y', None)
            for attr in ('edge_index_p', 'edge_weight_p', 'edge_index_n', 'edge_weight_n'):
                if hasattr(data, attr):
                    setattr(data_prompted, attr, getattr(data, attr))
        data_prompted.x = prompted_x
        z, _, _, _ = self.encoder(data_prompted, pi)
        return z

    def pretrain_step(self, data, pi, train_edges, train_labels, optimizer):
        self.encoder.train()
        self.prompt.train()
        self.edge_decoder.train()
        optimizer.zero_grad()
        z = self._encode_with_gpf(data, pi)
        logits = self.edge_decoder(z, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    def forward(self, data, pi=None):
        z = self._encode_with_gpf(data, pi)
        out = self.answering(z)
        log_prob = F.log_softmax(out, dim=1)
        pred     = out.argmax(dim=1)
        prob     = F.softmax(out, dim=1)
        return z, log_prob, pred, prob


class SignedAllInOneWrapper(nn.Module):
    """All-in-One prompt (gate + additive node token) with a selectable backbone."""

    def __init__(self, backbone: str, num_features: int, pi_dim: int,
                 hidden: int, label_dim: int, tda_type: str,
                 num_layers: int, dropout: float, args, device):
        super().__init__()
        self.device = device
        self.encoder, self.embed_dim = _make_signed_encoder(
            backbone=backbone,
            num_features=num_features,
            pi_dim=pi_dim,
            hidden=hidden,
            tda_type=tda_type,
            args=args,
        )
        self.node_prompt = nn.Parameter(torch.empty(1, self.embed_dim))
        nn.init.xavier_uniform_(self.node_prompt)
        self.answering = nn.Linear(self.embed_dim, label_dim)
        self.edge_decoder = EdgeSignDecoder(self.embed_dim)

    def prepare_graph(self, data):
        _prepare_graph_dependent_encoder(self.encoder, data, self.device)

    def encode(self, real, imag, edge_index, edge_weight, pi=None):
        class _D:
            pass
        d = _D(); d.x = real; d.edge_index = edge_index; d.edge_weight = edge_weight
        self.prepare_graph(d)
        if hasattr(self.encoder, 'encode'):
            return self.encoder.encode(real, imag, edge_index, edge_weight, pi)
        z, _, _, _ = self.encoder(d, pi)
        return z

    def pretrain_step(self, data, pi, train_edges, train_labels, optimizer):
        self.encoder.train(); self.edge_decoder.train()
        optimizer.zero_grad()
        z = self.encode(data.x, data.x, data.edge_index, data.edge_weight, pi)
        logits = self.edge_decoder(z, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
        loss.backward(); optimizer.step()
        return loss.item()

    def forward(self, data, pi=None):
        z = self.encode(data.x, data.x, data.edge_index, data.edge_weight, pi)
        gate = torch.sigmoid((z * self.node_prompt).sum(dim=1, keepdim=True))
        z_prompted = z + gate * self.node_prompt
        out = self.answering(z_prompted)
        log_prob = F.log_softmax(out, dim=1)
        return z_prompted, log_prob, out.argmax(dim=1), F.softmax(out, dim=1)


class SAMGPTWrapper(nn.Module):
    """Unsigned GCN-based SAMGPT backbone for signed graphs.

    Uses absolute values of edge weights as the unsigned adjacency.
    Interface matches SignedGPPT / SignedGprompt / SignedGPF:
        forward(data, pi=None)  -> (z, log_prob, pred, prob)
        pretrain_step(data, pi, train_edges, train_labels, optimizer) -> loss
        edge_decoder  (EdgeSignDecoder instance)
    """

    def __init__(self, num_features: int, hidden: int, label_dim: int,
                 num_layers: int, dropout: float, args, device):
        super().__init__()
        self.device    = device
        self.embed_dim = hidden

        try:
            from toposign.models.backbones.samgpt.gcnlayers import GcnLayers as _GcnLayers
        except ImportError as e:
            raise ImportError(
                "Could not import GcnLayers from toposign.models.backbones.samgpt. "
                f"Original error: {e}"
            )

        self.gcn        = _GcnLayers(num_features, hidden, num_layers, dropout)
        self.answering  = nn.Linear(hidden, label_dim)
        self.edge_decoder = EdgeSignDecoder(hidden)

    def _edge_index_to_sparse_adj(self, edge_index: torch.Tensor,
                                  edge_weight: Optional[torch.Tensor],
                                  num_nodes: int) -> torch.Tensor:
        if edge_weight is not None:
            vals = edge_weight.abs().float()
        else:
            vals = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
        adj = torch.sparse_coo_tensor(
            edge_index, vals, (num_nodes, num_nodes),
            device=edge_index.device,
        ).coalesce()
        return adj

    def encode(self, real: torch.Tensor, imag: torch.Tensor,
               edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
               pi=None) -> torch.Tensor:
        num_nodes = real.size(0)
        adj = self._edge_index_to_sparse_adj(edge_index, edge_weight, num_nodes)
        z = self.gcn(real, adj, sparse=True, LP=False)
        return z.squeeze(0)

    def pretrain_step(self, data, pi, train_edges, train_labels, optimizer):
        self.gcn.train()
        self.edge_decoder.train()
        optimizer.zero_grad()
        z = self.encode(data.x, data.x, data.edge_index, data.edge_weight, pi)
        logits = self.edge_decoder(z, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    def forward(self, data, pi=None):
        z = self.encode(data.x, data.x, data.edge_index, data.edge_weight, pi)
        out      = self.answering(z)
        log_prob = F.log_softmax(out, dim=1)
        pred     = out.argmax(dim=1)
        prob     = F.softmax(out, dim=1)
        return z, log_prob, pred, prob


def build_gfm(prompt_method: str, backbone: str,
              num_features: int, pi_dim: int,
              hidden: int, label_dim: int,
              tda_type: str, args, device) -> nn.Module:
    """Build a GFM prompt model with the specified signed backbone."""
    if prompt_method == 'SAMGPT':
        return SAMGPTWrapper(
            num_features=num_features,
            hidden=hidden,
            label_dim=label_dim,
            num_layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.0),
            args=args,
            device=device,
        )
    if prompt_method == 'All-in-one':
        encoder_backbone = 'GCN' if backbone == 'All-in-one' else backbone
        return SignedAllInOneWrapper(
            backbone=encoder_backbone,
            num_features=num_features,
            pi_dim=pi_dim,
            hidden=hidden,
            label_dim=label_dim,
            tda_type=tda_type,
            num_layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.0),
            args=args,
            device=device,
        )
    if prompt_method == 'TopoDIG':
        backbone = 'TopoDIG'
        prompt_method = 'GPPT'
    kwargs = dict(
        backbone=backbone, num_features=num_features, pi_dim=pi_dim,
        hidden=hidden, label_dim=label_dim,
        tda_type=tda_type, args=args, device=device,
    )
    if prompt_method == 'GPPT':
        center_num = getattr(args, 'center_num', label_dim * 2)
        return SignedGPPT(center_num=center_num, **kwargs)
    elif prompt_method == 'Gprompt':
        return SignedGprompt(**kwargs)
    elif prompt_method == 'GPF':
        return SignedGPF(**kwargs)
    else:
        raise ValueError(f"Unknown GFM prompt method: {prompt_method}")
