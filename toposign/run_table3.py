"""
Table 3: Filtration ablation.
Variants (MSGNN backbone):
  MSGNN_only    : --tda_type none
  Topo_only     : topological features + TPL projection only
  MSGNN+posTopo : structural branch + positive-subgraph topological features
  TopoMSGNN     : structural branch + signed topological features (ours)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import torch
import numpy as np
from datetime import datetime
from copy import deepcopy

from toposign.parser import parse_args
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    ALL_EVAL_DATASETS, SDSBM_DATA_DIR, PI_TENSOR_DIR, POSITIVE_PI_TENSOR_DIR, RESULTS_DIR,
    split_real_run_dataset, split_sdsbm_run_dataset, resolve_pi_tensor_path,
)
from toposign.configs.seed_config import seed_everything
from toposign.data.features import set_weighted_signed_input_features
from toposign.data.split import get_node_split
from toposign.models.gfm import build_gfm
from toposign.runner import (
    append_jsonl, load_curve_record, run_per_seed_lr_search, should_skip_result,
    ari_summary_metadata,
)
from toposign.run_table1 import pretrain_model, finetune_and_eval, _run_finetune

from torch_geometric_signed_directed.utils import in_out_degree
from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data


ABLATION_VARIANTS = {
    'SGE_only':    {'tda_type': 'none',     'use_structural': True},
    'Topo_only':   {'tda_type': 'signed',   'use_structural': False},
    'SGE+posTopo': {'tda_type': 'positive', 'use_structural': True},
    'TopoSIGN':    {'tda_type': 'signed',   'use_structural': True},
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


def load_pi(dataset_name: str, num_nodes: int, tda_type: str = 'signed', pixel_size: float = None):
    base_real, _ = split_real_run_dataset(dataset_name)
    pi_name = base_real if base_real in REAL_DATASETS else dataset_name
    path = resolve_pi_tensor_path(pi_name, tda_type, pixel_size=pixel_size)
    if os.path.exists(path):
        pi = torch.load(path, map_location='cpu')
        return pi.view(num_nodes, -1)
    return None




def _variant_items(selector: str):
    items = list(ABLATION_VARIANTS.items())
    if selector == 'nonpositive':
        return [item for item in items if item[0] != 'SGE+posTopo']
    if selector == 'positive':
        return [item for item in items if item[0] == 'SGE+posTopo']
    if selector != 'all':
        return [item for item in items if item[0] == selector]
    return items


def run_table3(args):
    if getattr(args, 'hparam_search', None) is None:
        args.hparam_search = True
    results_dir = args.results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'table3_filtration_ablation.jsonl')

    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    if args.dataset == 'all':
        datasets = list(ALL_EVAL_DATASETS)
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        data, run_id = load_dataset(dataset_name)
        data = set_weighted_signed_input_features(data)
        data = get_node_split(data, dataset_name, run_id)
        data = set_weighted_signed_input_features(data)

        num_nodes    = int(data.edge_index.max().item()) + 1
        num_features = data.x.size(1)
        label_dim    = int(data.y.max().item()) + 1
        signed_pi    = load_pi(dataset_name, num_nodes, 'signed', pixel_size=args.pixel_size)
        positive_pi  = load_pi(dataset_name, num_nodes, 'positive', pixel_size=args.pixel_size)

        req_backbone = getattr(args, 'backbone', 'TopoSIGN') or 'TopoSIGN'
        _variant_display = {
            'SGE_only':    f'{req_backbone}_only'    if req_backbone != 'TopoSIGN' else 'MSGNN_only',
            'Topo_only':   'Topo_only',
            'SGE+posTopo': f'{req_backbone}+posTopo' if req_backbone != 'TopoSIGN' else 'MSGNN+posTopo',
            'TopoSIGN':    f'Topo{req_backbone}'     if req_backbone != 'TopoSIGN' else 'TopoMSGNN',
        }

        for variant_name, variant_cfg in _variant_items(args.variant):
            tda_type       = variant_cfg['tda_type']
            use_structural = variant_cfg['use_structural']
            gfm_backbone = 'TopoSIGN' if not use_structural else req_backbone
            display_name  = _variant_display.get(variant_name, variant_name)
            run_pi         = positive_pi if tda_type == 'positive' else (
                signed_pi if tda_type == 'signed' else None
            )
            if tda_type != 'none' and run_pi is None:
                raise FileNotFoundError(
                    f"{tda_type} PI tensor required for {variant_name} on {dataset_name}. "
                    f"Run toposign/experiments/compute_all_pi.sh --tda_type {tda_type} first."
                )
            eff_pi_dim = run_pi.size(1) if run_pi is not None and tda_type != 'none' else 0

            print(f"\n[Table3] dataset={dataset_name}  variant={display_name}  backbone={gfm_backbone}")
            table_name = 'table3_filtration_ablation'

            runner_args = deepcopy(args)
            runner_args.tda_type       = tda_type
            runner_args.use_structural = use_structural
            runner_args.no_structural  = not use_structural

            if should_skip_result(runner_args, results_dir, table_name, display_name, dataset_name):
                continue

            aris = []
            for seed in runner_args.seeds:
                if not runner_args.force:
                    cached = load_curve_record(
                        results_dir, table_name, display_name, dataset_name, seed,
                        phase='finetune', args=runner_args,
                    )
                    if cached is not None:
                        ari = float(cached['test_ari'])
                        aris.append(ari)
                        print(f"  seed={seed}  ARI={ari:.3f}  restored from curve checkpoint")
                        continue

                seed_everything(seed)
                model = build_gfm(
                    prompt_method='GPPT',
                    backbone=gfm_backbone,
                    num_features=num_features,
                    pi_dim=eff_pi_dim,
                    hidden=runner_args.hidden,
                    label_dim=label_dim,
                    tda_type=tda_type,
                    args=runner_args,
                    device=device,
                ).to(device)
                model = pretrain_model(
                    model, data, run_pi, None, runner_args, device,
                    table=table_name, method=display_name, dataset=dataset_name,
                    seed=seed, results_dir=results_dir,
                )
                if getattr(runner_args, 'hparam_search', False):
                    ari, selected_lr = run_per_seed_lr_search(
                        pretrained_model=model,
                        data=data,
                        pi=run_pi,
                        args=runner_args,
                        device=device,
                        run_finetune_fn=_run_finetune,
                        results_dir=results_dir,
                        table=table_name,
                        method=display_name,
                        dataset=dataset_name,
                        seed=seed,
                    )
                    print(f"  seed={seed}  selected_lr={selected_lr:g}  ARI={ari:.3f}")
                else:
                    ari = finetune_and_eval(
                        model, data, run_pi, runner_args, device,
                        table=table_name, method=display_name, dataset=dataset_name,
                        seed=seed, results_dir=results_dir,
                    )
                    print(f"  seed={seed}  ARI={ari:.3f}")
                aris.append(ari)

            mean_ari = float(np.mean(aris))
            std_ari  = float(np.std(aris))
            print(f"  Final ARI: {mean_ari:.3f} ± {std_ari:.3f}")

            record = {
                'method': display_name,
                'dataset': dataset_name,
                'filtration_variant': display_name,
                'tda_type': tda_type,
                'use_structural': use_structural,
                'mean_ari': round(mean_ari, 3),
                'std_ari': round(std_ari, 3),
                **ari_summary_metadata(runner_args.seeds, aris),
                'debug': bool(runner_args.debug),
                'timestamp': datetime.now().isoformat(),
            }
            append_jsonl(results_path, record)


if __name__ == '__main__':
    args = parse_args()
    if args.results_dir is None:
        args.results_dir = RESULTS_DIR
    run_table3(args)
