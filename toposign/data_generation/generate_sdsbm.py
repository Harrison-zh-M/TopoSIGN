"""Generate SDSBM datasets for all SDSBM_SETTINGS x 5 run_ids."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pickle
import torch

from torch_geometric_signed_directed.data.signed.SignedData import SignedData
from torch_geometric_signed_directed.utils.general.in_out_degree import in_out_degree
from torch_geometric_signed_directed.utils.general.extract_network import extract_network
from toposign.models.backbones.msgnn.utils.SDSBM import SDSBM
from toposign.configs.dataset_config import SDSBM_SETTINGS, SDSBM_DATA_DIR, get_F_matrix
from toposign.configs.seed_config import sdsbm_generation_seed


def generate_instance(setting_name: str, run_id: int, force: bool = False) -> None:
    out_dir = os.path.join(SDSBM_DATA_DIR, setting_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'run{run_id}.pkl')
    if os.path.exists(out_path) and not force:
        print(f"Skip {setting_name}/run{run_id} (already exists)")
        return

    cfg = SDSBM_SETTINGS[setting_name]
    seed = sdsbm_generation_seed(run_id)
    F = get_F_matrix(cfg['dataset_type'], cfg['gamma'])

    sparse_A, labels_array = SDSBM(
        N=cfg['N'],
        K=cfg['K'],
        p=cfg['p'],
        F=F,
        size_ratio=cfg['size_ratio'],
        eta=cfg['eta'],
        seed=seed,
    )

    sparse_A, labels_array = extract_network(sparse_A, labels_array)
    data = SignedData(A=sparse_A, y=torch.LongTensor(labels_array))

    num_nodes = int(data.edge_index.max().item()) + 1
    data.x = in_out_degree(
        data.edge_index, size=num_nodes, signed=True, edge_weight=data.edge_weight
    )

    tmp_path = f'{out_path}.{os.getpid()}.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp_path, out_path)
    print(f"Saved {setting_name}/run{run_id} -> {out_path}")


def generate_all(force: bool = False, settings: list = None) -> None:
    for setting_name in (settings or list(SDSBM_SETTINGS.keys())):
        for run_id in range(5):
            generate_instance(setting_name, run_id, force=force)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Regenerate datasets even when pickle files already exist')
    parser.add_argument('--settings', nargs='+', default=None,
                        help='Which settings to generate (default: all four settings)')
    args = parser.parse_args()
    generate_all(force=args.force, settings=args.settings)
