"""
Table 1: Link sign prediction pre-training + node clustering fine-tuning.
Methods: TopoMSGNN, TopoSSSNET plus original prompt-learning baselines.
"""

import sys, os
from copy import deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from sklearn.metrics import adjusted_rand_score

from toposign.parser import parse_args
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    ALL_EVAL_DATASETS, SDSBM_DATA_DIR, PI_TENSOR_DIR, POSITIVE_PI_TENSOR_DIR, RESULTS_DIR,
    PRETRAIN_CONFIG, HYPERPARAMETER_GRID,
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
    run_per_seed_lr_search, ari_summary_metadata,
)

from torch_geometric_signed_directed.utils import in_out_degree
from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data


NOT_IMPLEMENTED = []

TABLE = 'table1_pretrain_finetune'

TABLE1_METHODS = {
    'TopoMSGNN':  ('TopoSIGN', 'GPPT'),
    'TopoSSSNET': ('SSSNET',   'GPPT'),
    'GPPT':       ('GCN',       'GPPT'),
    'Gprompt':    ('GCN',       'Gprompt'),
    'GPF':        ('GCN',       'GPF'),
    'All-in-one': ('All-in-one', 'All-in-one'),
    'SAMGPT':     ('SAMGPT',    None),
    'TopoDIG':    ('TopoDIG',   'GPPT'),
}


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


def load_pi(dataset_name: str, num_nodes: int, tda_type: str = 'signed', pixel_size: float = None) -> torch.Tensor:
    base_real, _ = split_real_run_dataset(dataset_name)
    pi_name = base_real if base_real in REAL_DATASETS else dataset_name
    path = resolve_pi_tensor_path(pi_name, tda_type, pixel_size=pixel_size)
    if os.path.exists(path):
        pi = torch.load(path, map_location='cpu')
        return pi.view(num_nodes, -1)
    return None


def pretrain_model(model, data, pi, link_data, args, device, *,
                   table='', method='', dataset='', seed=0, results_dir=''):
    """Pre-train with BCE on link sign using all 10 folds."""
    t_start = time.perf_counter()
    pretrain_lr = getattr(args, 'pretrain_lr', PRETRAIN_CONFIG['lr'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=pretrain_lr, weight_decay=args.weight_decay
    )

    real        = data.x.to(device)
    imag        = data.x.to(device)
    edge_index  = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
    pi_dev      = pi.to(device) if pi is not None else None

    train_edges, train_labels = observed_sign_edges(data, device)

    pretrainer = LinkSignPretrainer(model, optimizer, device)
    _pe = getattr(args, 'pretrain_epochs', None)
    pretrain_epochs = PRETRAIN_CONFIG["epochs"] if _pe is None else _pe
    early_stopping = PRETRAIN_CONFIG['early_stopping']
    best_loss = float('inf')
    patience_counter = 0

    train_loss_curve = []
    for epoch in range(pretrain_epochs):
        batch_edges, batch_labels = sample_link_batch(
            train_edges, train_labels, getattr(args, 'pretrain_batch_size', 64)
        )
        if hasattr(model, 'pretrain_step'):
            _data = _make_data_obj(real, imag, edge_index, edge_weight, data)
            loss = model.pretrain_step(_data, pi_dev, batch_edges, batch_labels, optimizer)
        else:
            loss = pretrainer.train_epoch(
                real, imag, edge_index, edge_weight,
                batch_edges, batch_labels, pi_dev
            )
        train_loss_curve.append(round(loss, 6))
        if loss < best_loss - 1e-8:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if args.debug and epoch >= 1:
            break
        if patience_counter >= early_stopping:
            break

    elapsed = time.perf_counter() - t_start
    save_curve_file(
        results_dir=results_dir,
        table=table, method=method, dataset=dataset, seed=seed,
        epochs_run=len(train_loss_curve),
        elapsed_sec=elapsed,
        train_loss=train_loss_curve,
        val_ari_curve=[],
        best_val_ari=0.0,
        test_ari=0.0,
        phase='pretrain',
        debug=getattr(args, 'debug', False),
        config=training_config_from_args(args),
    )
    return model


def _run_finetune(model, data, pi, args, device):
    """Core fine-tuning loop; returns (test_ari, val_ari, train_loss_curve, val_ari_curve, elapsed_sec).

    Does not save any files — callers are responsible for logging.
    Uses args.epochs and args.early_stopping for the loop budget.
    """
    t_start = time.perf_counter()
    optimizer = torch.optim.Adam(
        prompt_tuning_parameters(model), lr=prompt_tuning_lr(model, args),
        weight_decay=args.weight_decay,
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
        train_loss_curve.append(round(float(loss.item()), 6))

        model.eval()
        with torch.no_grad():
            _data = _make_data_obj(real, imag, edge_index, edge_weight, data)
            _, _, pred_val, _ = sanitize_model_output(*model(_data, pi_dev))

        val_ari = float(adjusted_rand_score(
            y[val_mask].cpu().numpy(), pred_val[val_mask].cpu().numpy()
        ))
        val_ari_curve.append(round(val_ari, 6))
        if val_ari > best_val_ari:
            best_val_ari  = val_ari
            best_test_ari = float(adjusted_rand_score(
                y[test_mask].cpu().numpy(), pred_val[test_mask].cpu().numpy()
            ))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping:
            break
        if getattr(args, 'debug', False) and epoch >= 1:
            break

    elapsed = time.perf_counter() - t_start
    return best_test_ari, best_val_ari, train_loss_curve, val_ari_curve, elapsed


def finetune_and_eval(model, data, pi, args, device, *,
                      table='', method='', dataset='', seed=0, results_dir='') -> float:
    """Fine-tune prompt head on seed nodes, evaluate test ARI."""
    test_ari, val_ari, train_loss, val_ari_curve, elapsed = _run_finetune(
        model, data, pi, args, device
    )
    save_curve_file(
        results_dir=results_dir,
        table=table, method=method, dataset=dataset, seed=seed,
        epochs_run=len(train_loss),
        elapsed_sec=elapsed,
        train_loss=train_loss,
        val_ari_curve=val_ari_curve,
        best_val_ari=val_ari,
        test_ari=test_ari,
        phase='finetune',
        debug=getattr(args, 'debug', False),
        config=training_config_from_args(args),
    )
    return test_ari


def run_table1(args):
    if getattr(args, 'hparam_search', None) is None:
        args.hparam_search = True
    results_dir = args.results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'table1_pretrain_finetune.jsonl')

    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    if args.dataset == 'all':
        datasets = list(ALL_EVAL_DATASETS)
    else:
        datasets = [args.dataset]

    method_name = args.method
    if method_name in NOT_IMPLEMENTED:
        print(f"Method {method_name} not implemented. Skipping.")
        return

    if method_name not in TABLE1_METHODS and method_name != 'all':
        print(f"Method {method_name} not in Table 1 methods. Use run_table2.py instead.")
        return

    if method_name == 'all':
        methods_to_run = list(TABLE1_METHODS.keys())
    else:
        methods_to_run = [method_name]

    for dataset_name in datasets:
        data, run_id = load_dataset(dataset_name)
        data = set_weighted_signed_input_features(data)
        data = get_node_split(data, dataset_name, run_id)
        data = set_weighted_signed_input_features(data)
        link_data = None

        num_nodes    = int(data.edge_index.max().item()) + 1
        num_features = data.x.size(1)
        label_dim    = int(data.y.max().item()) + 1
        signed_pi    = load_pi(dataset_name, num_nodes, 'signed', pixel_size=args.pixel_size)
        positive_pi  = load_pi(dataset_name, num_nodes, 'positive', pixel_size=args.pixel_size)

        for method_name in methods_to_run:
            backbone, prompt_method = TABLE1_METHODS[method_name]
            uses_tda = backbone in ('TopoSIGN', 'TopoDIG', 'SSSNET')
            method_tda_type = 'positive' if method_name == 'TopoDIG' else (
                'signed' if backbone in ('TopoSIGN', 'SSSNET') else 'none'
            )
            pi = positive_pi if method_tda_type == 'positive' else (
                signed_pi if method_tda_type == 'signed' else None
            )
            if method_name == 'TopoDIG' and pi is None:
                raise FileNotFoundError(
                    f"Positive-only PI tensor required for TopoDIG on {dataset_name}. "
                    "Run toposign/experiments/compute_all_pi.sh --tda_type positive first."
                )
            pi_dim = pi.size(1) if pi is not None else 0
            print(f"\n[Table1] dataset={dataset_name}  method={method_name}")
            if should_skip_result(args, results_dir, TABLE, method_name, dataset_name):
                continue

            def _build_model():
                if backbone == 'SAMGPT' or prompt_method is None:
                    return build_gfm(
                        prompt_method='SAMGPT', backbone='SAMGPT',
                        num_features=num_features, pi_dim=0,
                        hidden=args.hidden, label_dim=label_dim,
                        tda_type='none', args=args, device=device,
                    ).to(device)
                return build_gfm(
                    prompt_method=prompt_method, backbone=backbone,
                    num_features=num_features,
                    pi_dim=pi_dim if uses_tda and method_tda_type != 'none' else 0,
                    hidden=args.hidden, label_dim=label_dim,
                    tda_type=method_tda_type if uses_tda else 'none',
                    args=args, device=device,
                ).to(device)

            aris = []
            for seed in args.seeds:
                if not args.force:
                    cached = load_curve_record(
                        results_dir, TABLE, method_name, dataset_name, seed,
                        phase='finetune', args=args,
                    )
                    if cached is not None:
                        ari = float(cached['test_ari'])
                        aris.append(ari)
                        print(f"  seed={seed}  ARI={ari:.3f}  [cached]")
                        continue

                seed_everything(seed)
                model = _build_model()
                model = pretrain_model(
                    model, data, pi, link_data, args, device,
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
    run_table1(args)
