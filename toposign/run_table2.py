"""Table 2: Node clustering — pure signed GNN backbones vs backbone+Topo variants.

Methods: SSSNET, SigMaNet, MSGNN, DSGC (no PI); same four with signed PI concatenated.
Loss:     finite NLL loss on train_mask (10% labeled nodes).
Metric:   ARI on test_mask (~80% nodes).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import torch
import torch.nn.functional as F
from datetime import datetime

from toposign.parser import parse_args
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    ALL_EVAL_DATASETS, SDSBM_DATA_DIR, PI_TENSOR_DIR, RESULTS_DIR, TABLE2_CONFIG,
    split_real_run_dataset, split_sdsbm_run_dataset, resolve_pi_tensor_path,
)
from toposign.data.features import set_weighted_signed_input_features
from toposign.data.split import get_node_split
from toposign.models.baselines import build_baseline
from toposign.models.toposign import TPL
from toposign.runner import (
    NodeClusteringRunner, save_result, should_skip_result, ari_summary_metadata,
)

from torch_geometric_signed_directed.utils import in_out_degree
from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data

TABLE = 'table2_node_clustering'
TABLE2_METHODS = [
    'SSSNET', 'SigMaNet', 'MSGNN', 'DSGC',
    'DSGC+Topo', 'SigMaNet+Topo', 'MSGNN+Topo', 'SSSNET+Topo',
]


def load_dataset(dataset_name: str):
    base_real, real_run_id = split_real_run_dataset(dataset_name)
    if base_real in REAL_DATASETS:
        data = load_signed_real_data(dataset=base_real, root='data/')
        if data.x is None:
            num_nodes = int(data.edge_index.max().item()) + 1
            data.x = in_out_degree(
                data.edge_index, size=num_nodes,
                signed=True, edge_weight=data.edge_weight,
            )
        return data, real_run_id if real_run_id is not None else 0

    setting_name, run_id = split_sdsbm_run_dataset(dataset_name)
    if run_id is not None:
        path = os.path.join(SDSBM_DATA_DIR, setting_name, f'run{run_id}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data, run_id
    elif dataset_name in SDSBM_SETTINGS:
        path = os.path.join(SDSBM_DATA_DIR, dataset_name, 'run0.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data, 0
    raise ValueError(f'Unknown dataset: {dataset_name}')


def load_pi(dataset_name: str, pi_tensor_dir: str = None, tda_type: str = 'signed',
            pixel_size: float = None):
    base_real, _ = split_real_run_dataset(dataset_name)
    pi_name = base_real if base_real in REAL_DATASETS else dataset_name
    path = resolve_pi_tensor_path(pi_name, tda_type, pi_tensor_dir, pixel_size=pixel_size)
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    return None


def _make_baseline_cls(method: str, original_data):
    """Return a class whose __init__ accepts (num_features, hidden, label_dim, args)."""

    class _BaselineWrapper(torch.nn.Module):
        def __init__(self, num_features, hidden, label_dim, args):
            super().__init__()
            self._inner = build_baseline(method, num_features, hidden, label_dim, args)
            if hasattr(self._inner, 'set_laplacian'):
                self.set_laplacian = self._inner.set_laplacian

        def forward(self, data, pi=None):
            return self._inner(data, pi)

        def parameters(self, recurse=True):
            return self._inner.parameters(recurse)

    _BaselineWrapper.__name__ = f'{method}Wrapper'
    return _BaselineWrapper


def _make_topo_baseline_cls(method: str):
    """Two-branch model mirroring TopoSIGN: Z = [Z_G || Z_T] → fused classifier.

    Z_G comes from the signed backbone (structural branch).
    Z_T comes from TPL applied to the signed PI tensor (topological branch).
    A fresh Conv1d classifier is trained on the concatenated representation.
    """

    class _TopoBackboneWrapper(torch.nn.Module):
        def __init__(self, num_features, hidden, label_dim, pi_dim, topo_hidden, args):
            super().__init__()
            self._backbone = build_baseline(method, num_features, hidden, label_dim, args)
            if hasattr(self._backbone, 'set_laplacian'):
                self.set_laplacian = self._backbone.set_laplacian

            self._tpl = TPL(pi_dim, topo_hidden, dropout=args.dropout)

            backbone_dim = getattr(self._backbone, 'embed_dim', 2 * hidden)
            self._classifier = torch.nn.Conv1d(backbone_dim + topo_hidden, label_dim, kernel_size=1)
            self._dropout_val = args.dropout

        def forward(self, data, pi=None):
            z_g, _, _, _ = self._backbone(data, None)

            if pi is not None:
                z_t = self._tpl(pi.view(z_g.size(0), -1).to(z_g.device))
                z = torch.cat([z_g, z_t], dim=-1)
            else:
                z = z_g

            if self._dropout_val > 0:
                z = F.dropout(z, self._dropout_val, training=self.training)

            logits = self._classifier(z.unsqueeze(0).permute(0, 2, 1))
            log_prob = F.log_softmax(logits, dim=1)[0].t()
            pred = log_prob.argmax(dim=1)
            prob = log_prob.exp()
            return F.normalize(z), log_prob, pred, prob

    _TopoBackboneWrapper.__name__ = f'{method}TopoWrapper'
    return _TopoBackboneWrapper


def run_table2(args):
    if getattr(args, 'hparam_search', None) is None:
        args.hparam_search = False
    results_dir = args.results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    if not args.debug:
        args.epochs = TABLE2_CONFIG['epochs']
        args.early_stopping = TABLE2_CONFIG['early_stopping']

    if args.dataset == 'all':
        datasets = list(ALL_EVAL_DATASETS)
    else:
        datasets = [args.dataset]

    methods = [args.method] if args.method != 'all' else TABLE2_METHODS

    for dataset_name in datasets:
        data, run_id = load_dataset(dataset_name)
        data = set_weighted_signed_input_features(data)
        data = get_node_split(data, dataset_name, run_id)
        data = set_weighted_signed_input_features(data)
        pi   = load_pi(dataset_name, pixel_size=args.pixel_size)

        num_nodes    = int(data.edge_index.max().item()) + 1
        num_features = data.x.size(1)
        label_dim    = int(data.y.max().item()) + 1
        pi_dim       = pi.view(num_nodes, -1).size(1) if pi is not None else 0

        for method in methods:
            print(f'\n[{TABLE}] dataset={dataset_name}  method={method}')
            if should_skip_result(args, results_dir, TABLE, method, dataset_name):
                continue

            if method.endswith('+Topo'):
                base_method = method[:-5]
                model_cls    = _make_topo_baseline_cls(base_method)
                model_kwargs = dict(
                    num_features=num_features,
                    hidden=args.hidden,
                    label_dim=label_dim,
                    pi_dim=pi_dim,
                    topo_hidden=getattr(args, 'topo_hidden', args.hidden),
                    args=args,
                )
                runner_pi = pi
            else:
                model_cls    = _make_baseline_cls(method, data)
                model_kwargs = dict(
                    num_features=num_features,
                    hidden=args.hidden,
                    label_dim=label_dim,
                    args=args,
                )
                runner_pi = None

            runner = NodeClusteringRunner(
                model_cls, model_kwargs, data, args,
                pi=runner_pi,
                table=TABLE,
                method=method,
                dataset=dataset_name,
                results_dir=results_dir,
            )
            mean_ari, std_ari = runner.run()
            save_result(
                results_dir, TABLE, method, dataset_name, mean_ari, std_ari,
                extra={
                    'debug': bool(args.debug),
                    **ari_summary_metadata(
                        getattr(runner, 'last_seeds', args.seeds),
                        getattr(runner, 'last_ari_values', []),
                    ),
                },
            )


if __name__ == '__main__':
    args = parse_args()
    if args.results_dir is None:
        args.results_dir = RESULTS_DIR
    run_table2(args)
