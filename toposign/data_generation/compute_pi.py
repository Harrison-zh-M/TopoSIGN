"""Compute persistence-image tensors for all datasets."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from typing import Iterable, List, Tuple
import torch

from torch_geometric_signed_directed.data.signed.SignedData import SignedData
from torch_geometric_signed_directed.utils.general.in_out_degree import in_out_degree

from tda_s_filtration import compute_tda_from_signed_data
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    SDSBM_DATA_DIR, PI_TENSOR_DIR, POSITIVE_PI_TENSOR_DIR, TDA_CONFIG,
    pi_tensor_filename,
)
from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data


DatasetJob = Tuple[str, str, str, int]


def _load_sdsbm(setting_name: str, run_id: int) -> SignedData:
    path = os.path.join(SDSBM_DATA_DIR, setting_name, f'run{run_id}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_real(dataset_name: str) -> SignedData:
    data = load_signed_real_data(dataset=dataset_name, root='data/')
    if data.x is None:
        num_nodes = int(data.edge_index.max().item()) + 1
        data.x = in_out_degree(
            data.edge_index, size=num_nodes, signed=True, edge_weight=data.edge_weight
        )
    return data


def _positive_only_data(data: SignedData) -> SignedData:
    """Return a shallow copy that keeps only positive signed edges."""
    out = copy.copy(data)
    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is None:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)
    mask = edge_weight > 0
    device = data.edge_index.device
    out.edge_index = data.edge_index[:, mask]
    out.edge_weight = edge_weight[mask].abs()
    out.edge_index_p = out.edge_index
    out.edge_weight_p = out.edge_weight
    out.edge_index_n = torch.empty((2, 0), dtype=torch.long, device=device)
    out.edge_weight_n = torch.empty((0,), dtype=out.edge_weight.dtype, device=out.edge_weight.device)
    return out


def compute_and_save(
    dataset_name: str,
    data: SignedData,
    *,
    tda_type: str = 'signed',
    pixel_size: float = None,
    force: bool = False,
) -> str:
    pi_dir = POSITIVE_PI_TENSOR_DIR if tda_type == 'positive' else PI_TENSOR_DIR
    os.makedirs(pi_dir, exist_ok=True)
    out_path = os.path.join(pi_dir, pi_tensor_filename(dataset_name, tda_type, pixel_size=pixel_size))
    if os.path.exists(out_path) and not force:
        return f"Skip {dataset_name} {tda_type} PI pixel_size={pixel_size or TDA_CONFIG['pixel_size']} (already exists)"
    if tda_type == 'positive':
        data = _positive_only_data(data)
    tda_config = dict(TDA_CONFIG)
    if pixel_size is not None:
        tda_config['pixel_size'] = pixel_size
    pi = compute_tda_from_signed_data(
        data,
        include_empty_images=(tda_type == 'positive'),
        **tda_config,
    )
    tmp_path = f'{out_path}.{os.getpid()}.tmp'
    torch.save(pi, tmp_path)
    os.replace(tmp_path, out_path)
    return (
        f"Saved {tda_type} PI tensor for {dataset_name} "
        f"pixel_size={tda_config['pixel_size']} -> {out_path}  shape={tuple(pi.shape)}"
    )


def _dataset_jobs(target: str) -> List[DatasetJob]:
    if target == 'all':
        return [
            (f'{setting_name}_run{run_id}', 'sdsbm', setting_name, run_id)
            for setting_name in SDSBM_SETTINGS
            for run_id in range(5)
        ] + [
            (dataset_name, 'real', dataset_name, -1)
            for dataset_name in REAL_DATASETS
        ]
    if target in REAL_DATASETS:
        return [(target, 'real', target, -1)]
    elif '_run' in target:
        setting_name, run_part = target.rsplit('_run', 1)
        run_id = int(run_part)
        return [(target, 'sdsbm', setting_name, run_id)]
    elif target in SDSBM_SETTINGS:
        return [
            (f'{target}_run{run_id}', 'sdsbm', target, run_id)
            for run_id in range(5)
        ]
    raise ValueError(f"Unknown target: {target}")


def _run_job(job: DatasetJob, tda_type: str = 'signed',
             pixel_size: float = None, force: bool = False) -> str:
    dataset_name, kind, key, run_id = job
    if kind == 'sdsbm':
        data = _load_sdsbm(key, run_id)
    elif kind == 'real':
        data = _load_real(key)
    else:
        raise ValueError(f'Unknown job kind: {kind}')
    return compute_and_save(
        dataset_name, data, tda_type=tda_type, pixel_size=pixel_size, force=force
    )


def _resolve_workers(workers: int, jobs: Iterable[DatasetJob]) -> int:
    jobs = list(jobs)
    if workers < 1:
        workers = os.cpu_count() or 1
    return max(1, min(workers, len(jobs)))


def main(target: str = 'all', workers: int = 1, tda_type: str = 'signed',
         pixel_size: float = None, force: bool = False) -> None:
    jobs = _dataset_jobs(target)
    workers = _resolve_workers(workers, jobs)

    if workers == 1:
        for job in jobs:
            print(_run_job(job, tda_type=tda_type, pixel_size=pixel_size, force=force), flush=True)
        return

    print(
        f"[compute_pi] Running {len(jobs)} {tda_type} dataset jobs "
        f"with {workers} workers. pixel_size={pixel_size or TDA_CONFIG['pixel_size']}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_job, job, tda_type, pixel_size, force): job[0]
            for job in jobs
        }
        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(f"PI computation failed for {dataset_name}") from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        help='all | settingN | settingN_runM | rainfall | sp1500')
    parser.add_argument('--workers', type=int, default=int(os.environ.get('PI_JOBS', '1')),
                        help='Number of parallel dataset PI workers. Use 0 for all CPUs.')
    parser.add_argument('--tda_type', choices=['signed', 'positive'], default='signed',
                        help='signed computes the default signed PI; positive keeps only positive edges.')
    parser.add_argument('--pixel_size', type=float, default=None,
                        help='Persistence-image pixel size. Defaults to TDA_CONFIG pixel_size.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute PI tensors even when output files already exist.')
    args = parser.parse_args()
    main(
        args.dataset,
        workers=args.workers,
        tda_type=args.tda_type,
        pixel_size=args.pixel_size,
        force=args.force,
    )
