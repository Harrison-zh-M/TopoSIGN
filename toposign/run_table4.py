"""
Table 4: Pre-training task ablation.
Tasks: SP, SP+DP, SP+3C, SP+4C, SP+5C.
Each variant pre-trains TopoSIGN with different task combinations, then
fine-tunes with GPPT for node clustering.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from copy import deepcopy
from torch_geometric.utils import negative_sampling

from toposign.parser import parse_args
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    SDSBM_DATA_DIR, PI_TENSOR_DIR, RESULTS_DIR, TABLE4_DATASETS,
    PRETRAIN_CONFIG,
    split_real_run_dataset, split_sdsbm_run_dataset, resolve_pi_tensor_path,
)
from toposign.configs.seed_config import seed_everything
from toposign.data.features import set_weighted_signed_input_features
from toposign.data.split import get_node_split
from toposign.models.gfm import build_gfm
from toposign.runner import (
    LinkSignPretrainer, _make_data_obj, append_jsonl,
    load_curve_record, observed_sign_edges, prompt_tuning_parameters,
    prompt_tuning_lr, sample_link_batch, save_curve_file, should_skip_result,
    training_config_from_args, sanitize_model_output, finite_nll_loss,
    finite_nll_loss_from_logits, run_per_seed_lr_search,
    ari_summary_metadata,
)
from toposign.run_table1 import _run_finetune

from torch_geometric_signed_directed.utils import in_out_degree
from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data
from sklearn.metrics import adjusted_rand_score


PRETRAIN_VARIANTS = ['SP+DP', 'SP+3C', 'SP+4C', 'SP+5C']

TABLE = 'table4_pretrain_task_ablation'


class PairPredictionHead(torch.nn.Module):
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = max(32, embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return self.mlp(torch.cat([src, dst], dim=-1))


def _model_embeddings(model, data, pi):
    if hasattr(model, 'encode'):
        return model.encode(data, pi)
    z, _, _, _ = model(data, pi)
    return z


def load_dataset(dataset_name: str):
    base_real, real_run_id = split_real_run_dataset(dataset_name)
    if base_real in REAL_DATASETS:
        data = load_signed_real_data(dataset=base_real, root='data/')
        if data.x is None:
            num_nodes = int(data.edge_index.max().item()) + 1
            data.x = in_out_degree(
                data.edge_index, size=num_nodes, signed=True, edge_weight=data.edge_weight
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
    raise ValueError(f"Unknown dataset: {dataset_name}")


def load_pi(dataset_name: str, num_nodes: int, tda_type: str = 'signed', pixel_size: float = None):
    base_real, _ = split_real_run_dataset(dataset_name)
    pi_name = base_real if base_real in REAL_DATASETS else dataset_name
    path = resolve_pi_tensor_path(pi_name, tda_type, pixel_size=pixel_size)
    if os.path.exists(path):
        pi = torch.load(path, map_location='cpu')
        return pi.view(num_nodes, -1)
    return None


def run_pretrain_variant(model, data, pi, link_data, tasks, args, device, *,
                         table='', method='', dataset='', seed=0, results_dir=''):
    """Pre-train with SP plus optional DP/3C/4C/5C sampled link objectives."""
    t_start = time.perf_counter()

    real        = data.x.to(device)
    imag        = data.x.to(device)
    edge_index  = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
    pi_dev      = pi.to(device) if pi is not None else None
    data_dev    = _make_data_obj(real, imag, edge_index, edge_weight, data)

    train_edges, train_labels = observed_sign_edges(data, device)

    with torch.no_grad():
        z0 = _model_embeddings(model, data_dev, pi_dev)
    embed_dim = z0.size(1)

    aux_heads = torch.nn.ModuleDict()
    if 'DP' in tasks:
        aux_heads['DP'] = PairPredictionHead(embed_dim, 1)
    for task in ('3C', '4C', '5C'):
        if task in tasks:
            aux_heads[task] = PairPredictionHead(embed_dim, int(task[0]))
    aux_heads.to(device)

    pretrain_lr = getattr(args, 'pretrain_lr', PRETRAIN_CONFIG.get('lr', args.lr))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aux_heads.parameters()),
        lr=pretrain_lr,
        weight_decay=args.weight_decay,
    )

    _pe = getattr(args, "pretrain_epochs", None)
    pretrain_epochs = PRETRAIN_CONFIG["epochs"] if _pe is None else _pe
    early_stopping = PRETRAIN_CONFIG['early_stopping']
    batch_size = getattr(args, 'pretrain_batch_size', 64)
    num_nodes = data.x.size(0)
    best_loss = float('inf')
    patience_counter = 0
    train_loss_curve = []

    for epoch in range(pretrain_epochs):
        model.train()
        aux_heads.train()
        optimizer.zero_grad()

        batch_edges, batch_labels = sample_link_batch(train_edges, train_labels, batch_size)
        z = _model_embeddings(model, data_dev, pi_dev)

        sp_logits = model.edge_decoder(z, batch_edges)
        loss = F.binary_cross_entropy_with_logits(sp_logits, batch_labels.float())

        if 'DP' in tasks:
            dp_edges = torch.cat([batch_edges, batch_edges.flip(0)], dim=1)
            dp_labels = torch.cat([
                torch.ones(batch_edges.size(1), device=device),
                torch.zeros(batch_edges.size(1), device=device),
            ])
            dp_logits = aux_heads['DP'](z, dp_edges).squeeze(-1)
            loss = loss + F.binary_cross_entropy_with_logits(dp_logits, dp_labels)

        if any(task in tasks for task in ('3C', '4C', '5C')):
            sign_labels = (batch_labels > 0.5).long()
            non_edges = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=batch_edges.size(1),
                method='sparse',
            ).to(device)

            if '3C' in tasks:
                cls_edges = torch.cat([batch_edges, non_edges], dim=1)
                cls_labels = torch.cat([
                    sign_labels,
                    torch.full((non_edges.size(1),), 2, dtype=torch.long, device=device),
                ])
                loss = loss + finite_nll_loss_from_logits(
                    aux_heads['3C'](z, cls_edges), cls_labels
                )

            if '4C' in tasks:
                cls_edges = torch.cat([batch_edges, batch_edges.flip(0)], dim=1)
                cls_labels = torch.cat([
                    sign_labels * 2 + 1,
                    sign_labels * 2,
                ])
                loss = loss + finite_nll_loss_from_logits(
                    aux_heads['4C'](z, cls_edges), cls_labels
                )

            if '5C' in tasks:
                cls_edges = torch.cat([batch_edges, batch_edges.flip(0), non_edges], dim=1)
                cls_labels = torch.cat([
                    sign_labels * 2 + 1,
                    sign_labels * 2,
                    torch.full((non_edges.size(1),), 4, dtype=torch.long, device=device),
                ])
                loss = loss + finite_nll_loss_from_logits(
                    aux_heads['5C'](z, cls_edges), cls_labels
                )

        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        train_loss_curve.append(round(loss_value, 6))
        if loss_value < best_loss - 1e-8:
            best_loss = loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if args.debug and epoch >= 1:
            break
        if patience_counter >= early_stopping:
            break

    save_curve_file(
        results_dir=results_dir,
        table=table, method=method, dataset=dataset, seed=seed,
        epochs_run=len(train_loss_curve),
        elapsed_sec=time.perf_counter() - t_start,
        train_loss=train_loss_curve,
        val_ari_curve=[],
        best_val_ari=0.0,
        test_ari=0.0,
        phase='pretrain',
        debug=getattr(args, 'debug', False),
        config=training_config_from_args(args),
    )
    return model


def finetune_and_eval(model, data, pi, args, device, *,
                      table='', method='', dataset='', seed=0, results_dir='') -> float:
    t_start = time.perf_counter()
    optimizer = torch.optim.Adam(
        prompt_tuning_parameters(model), lr=prompt_tuning_lr(model, args),
        weight_decay=args.weight_decay
    )

    real        = data.x.to(device)
    imag        = data.x.to(device)
    edge_index  = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
    y           = data.y.to(device)
    pi_dev      = pi.to(device) if pi is not None else None

    train_mask = data.train_mask[:, 0].to(device)
    val_mask   = data.val_mask[:, 0].to(device)
    test_mask  = data.test_mask[:, 0].to(device)

    if hasattr(model, 'init_prompt'):
        _data = _make_data_obj(real, imag, edge_index, edge_weight, data)
        _data.y = y
        model.init_prompt(_data, pi_dev, train_mask)

    best_val_ari  = -1.0
    best_test_ari = 0.0
    patience_counter = 0
    early_stopping = getattr(args, 'early_stopping', 400)

    train_loss_curve = []
    val_ari_curve    = []

    for epoch in range(args.epochs):
        model.train()
        _data = _make_data_obj(real, imag, edge_index, edge_weight, data)
        _data.y = y
        z, log_prob, pred, prob = sanitize_model_output(*model(_data, pi_dev))
        loss = finite_nll_loss(log_prob[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_curve.append(round(loss.item(), 6))

        model.eval()
        with torch.no_grad():
            _data = _make_data_obj(real, imag, edge_index, edge_weight, data)
            _, _, pred_val, _ = sanitize_model_output(*model(_data, pi_dev))

        val_ari = adjusted_rand_score(
            y[val_mask].cpu().numpy(), pred_val[val_mask].cpu().numpy()
        )
        val_ari_curve.append(round(float(val_ari), 6))
        if val_ari > best_val_ari:
            best_val_ari  = val_ari
            best_test_ari = adjusted_rand_score(
                y[test_mask].cpu().numpy(), pred_val[test_mask].cpu().numpy()
            )
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping:
            break

    elapsed = time.perf_counter() - t_start
    save_curve_file(
        results_dir=results_dir,
        table=table, method=method, dataset=dataset, seed=seed,
        epochs_run=len(train_loss_curve),
        elapsed_sec=elapsed,
        train_loss=train_loss_curve,
        val_ari_curve=val_ari_curve,
        best_val_ari=best_val_ari,
        test_ari=best_test_ari,
        phase='finetune',
        debug=getattr(args, 'debug', False),
        config=training_config_from_args(args),
    )
    return best_test_ari


def run_table4(args):
    if getattr(args, 'hparam_search', None) is None:
        args.hparam_search = True
    results_dir = args.results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'table4_pretrain_task_ablation.jsonl')

    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    if args.dataset == 'all':
        datasets = list(TABLE4_DATASETS)
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        data, run_id = load_dataset(dataset_name)
        data = set_weighted_signed_input_features(data)
        data = get_node_split(data, dataset_name, run_id)
        data = set_weighted_signed_input_features(data)
        link_data = None

        num_nodes    = int(data.edge_index.max().item()) + 1
        num_features = data.x.size(1)
        label_dim    = int(data.y.max().item()) + 1
        pi           = load_pi(dataset_name, num_nodes, pixel_size=args.pixel_size)
        pi_dim       = pi.size(1) if pi is not None else 0

        if args.variant == 'all':
            task_variants = list(PRETRAIN_VARIANTS)
        else:
            requested_variant = (args.variant
                                 .replace('TopoSIGN_pretrain_', '')
                                 .replace('TopoSSSNET_pretrain_', ''))
            task_variants = [requested_variant] if requested_variant in PRETRAIN_VARIANTS else []

        req_backbone = getattr(args, 'backbone', 'TopoSIGN') or 'TopoSIGN'
        method_prefix = f'Topo{req_backbone}' if req_backbone != 'TopoSIGN' else 'TopoMSGNN'

        for task_variant in task_variants:
            print(f"\n[Table4] dataset={dataset_name}  tasks={task_variant}  backbone={req_backbone}")
            tasks = task_variant.split('+')
            method_name = f'{method_prefix}_pretrain_{task_variant}'
            if should_skip_result(args, results_dir, TABLE, method_name, dataset_name):
                continue

            aris = []
            for seed in args.seeds:
                cached = None
                if not args.force:
                    cached = load_curve_record(
                        results_dir, TABLE, method_name, dataset_name, seed,
                        phase='finetune', args=args,
                    )
                if cached is not None:
                    ari = float(cached['test_ari'])
                    aris.append(ari)
                    print(f"  seed={seed}  ARI={ari:.3f}  restored from curve checkpoint")
                    continue

                seed_everything(seed)

                model = build_gfm(
                    prompt_method='GPPT',
                    backbone=req_backbone,
                    num_features=num_features,
                    pi_dim=pi_dim,
                    hidden=args.hidden,
                    label_dim=label_dim,
                    tda_type=args.tda_type,
                    args=args,
                    device=device,
                ).to(device)

                model = run_pretrain_variant(
                    model, data, pi, link_data, tasks, args, device,
                    table=TABLE, method=method_name, dataset=dataset_name,
                    seed=seed, results_dir=results_dir,
                )
                if getattr(args, 'hparam_search', False):
                    ari, selected_lr = run_per_seed_lr_search(
                        pretrained_model=model,
                        data=data,
                        pi=pi,
                        args=args,
                        device=device,
                        run_finetune_fn=_run_finetune,
                        results_dir=results_dir,
                        table=TABLE,
                        method=method_name,
                        dataset=dataset_name,
                        seed=seed,
                    )
                    print(f"  seed={seed}  selected_lr={selected_lr:g}  ARI={ari:.3f}")
                else:
                    ari = finetune_and_eval(
                        model, data, pi, args, device,
                        table=TABLE, method=method_name, dataset=dataset_name,
                        seed=seed, results_dir=results_dir,
                    )
                    print(f"  seed={seed}  ARI={ari:.3f}")
                aris.append(ari)

            mean_ari = float(np.mean(aris))
            std_ari  = float(np.std(aris))
            print(f"  Final ARI: {mean_ari:.3f} ± {std_ari:.3f}")

            record = {
                'method': method_name,
                'dataset': dataset_name,
                'pretrain_tasks': task_variant,
                'mean_ari': round(mean_ari, 3),
                'std_ari': round(std_ari, 3),
                **ari_summary_metadata(args.seeds, aris),
                'debug': bool(args.debug),
                'timestamp': datetime.now().isoformat(),
            }
            append_jsonl(results_path, record)


if __name__ == '__main__':
    args = parse_args()
    if args.results_dir is None:
        args.results_dir = RESULTS_DIR
    run_table4(args)
