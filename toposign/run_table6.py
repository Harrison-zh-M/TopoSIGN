"""
Table 6: Backbone ablation.
Methods: signed GNN backbones with Topo+GPPT plus prompt-learning variants
coupled to the SGE+Topo encoder.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
import time
import torch
import numpy as np
from datetime import datetime

from toposign.parser import parse_args
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    ALL_EVAL_DATASETS, TABLE6_DATASETS, SDSBM_DATA_DIR, PI_TENSOR_DIR, POSITIVE_PI_TENSOR_DIR,
    RESULTS_DIR, PRETRAIN_CONFIG,
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
from toposign.run_table1 import pretrain_model, finetune_and_eval, _run_finetune

from torch_geometric_signed_directed.utils import in_out_degree
from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data
from sklearn.metrics import adjusted_rand_score


BACKBONE_METHODS = ['SigMaNet', 'DSGC']

TABLE = 'table6_backbone_ablation'


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


def run_table6(args):
    if getattr(args, 'hparam_search', None) is None:
        args.hparam_search = True
    results_dir = args.results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'table6_backbone_ablation.jsonl')

    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    if args.dataset == 'all':
        datasets = list(TABLE6_DATASETS)
    else:
        datasets = [args.dataset]

    backbones = [args.backbone] if args.backbone and args.backbone in BACKBONE_METHODS else BACKBONE_METHODS

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

        for backbone in backbones:
            prompt_method = 'GPPT'
            encoder_backbone = backbone
            method_label = f'{backbone}+Topo+GPPT'
            pi = signed_pi
            pi_dim = pi.size(1) if pi is not None else 0
            method_tda_type = args.tda_type
            print(f"\n[Table6] dataset={dataset_name}  method={method_label}")
            if should_skip_result(args, results_dir, TABLE, method_label, dataset_name):
                continue

            aris = []
            for seed in args.seeds:
                cached = None
                if not args.force:
                    cached = load_curve_record(
                        results_dir, TABLE, method_label, dataset_name, seed,
                        phase='finetune', args=args,
                    )
                if cached is not None:
                    ari = float(cached['test_ari'])
                    aris.append(ari)
                    print(f"  seed={seed}  ARI={ari:.3f}  restored from curve checkpoint")
                    continue

                seed_everything(seed)

                model = build_gfm(
                    prompt_method=prompt_method,
                    backbone=encoder_backbone,
                    num_features=num_features,
                    pi_dim=pi_dim,
                    hidden=args.hidden,
                    label_dim=label_dim,
                    tda_type=method_tda_type,
                    args=args,
                    device=device,
                ).to(device)

                model = pretrain_model(
                    model, data, pi, link_data, args, device,
                    table=TABLE, method=method_label, dataset=dataset_name,
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
                        method=method_label,
                        dataset=dataset_name,
                        seed=seed,
                    )
                    print(f"  seed={seed}  selected_lr={selected_lr:g}  ARI={ari:.3f}")
                else:
                    ari = finetune_and_eval(
                        model, data, pi, args, device,
                        table=TABLE, method=method_label, dataset=dataset_name,
                        seed=seed, results_dir=results_dir,
                    )
                    print(f"  seed={seed}  ARI={ari:.3f}")
                aris.append(ari)

            mean_ari = float(np.mean(aris))
            std_ari  = float(np.std(aris))
            print(f"  Final ARI: {mean_ari:.3f} ± {std_ari:.3f}")

            record = {
                'method': method_label,
                'dataset': dataset_name,
                'backbone': encoder_backbone,
                'prompt_method': prompt_method,
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
    run_table6(args)
