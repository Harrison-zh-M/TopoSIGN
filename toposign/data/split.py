"""Cached node and link splits for all datasets."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pickle
import time
from contextlib import contextmanager

from torch_geometric_signed_directed.utils import link_class_split

from toposign.configs.dataset_config import SPLIT_CACHE_DIR, LINK_SPLIT, NODE_SPLIT
from toposign.configs.seed_config import link_split_seed, node_split_seed


def _fmt_ratio(value) -> str:
    return str(value).replace('.', 'p')


@contextmanager
def _cache_lock(cache_path: str, timeout: float = 600.0):
    """Cross-process lock for cache files written by parallel table jobs."""
    lock_dir = f'{cache_path}.lock'
    start = time.monotonic()
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f'Timed out waiting for cache lock: {lock_dir}')
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            os.rmdir(lock_dir)
        except FileNotFoundError:
            pass


def _atomic_pickle_dump(obj, cache_path: str) -> None:
    tmp_path = f'{cache_path}.{os.getpid()}.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, cache_path)


def get_node_split(data, dataset_name: str, run_id: int = None):
    """Return data with node split masks (train/val/test/seed). Caches result."""
    cache_dir = os.path.join(SPLIT_CACHE_DIR, 'node')
    os.makedirs(cache_dir, exist_ok=True)
    seed = node_split_seed(run_id) if run_id is not None else 0
    cache_name = (
        f'{dataset_name}_seed{seed}'
        f'_seedratio{_fmt_ratio(NODE_SPLIT["seed_ratio"])}'
        f'_valratio{_fmt_ratio(NODE_SPLIT["val_ratio"])}'
        '_datasplit1_v2.pkl'
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    with _cache_lock(cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        data.node_split(
            train_size_per_class=NODE_SPLIT['seed_ratio'],
            val_size_per_class=NODE_SPLIT['val_ratio'],
            seed_size_per_class=NODE_SPLIT['seed_ratio'],
            seed=[seed],
            data_split=1,
        )
        # seed_mask is set equal to train_mask (10% of all nodes).
        data.seed_mask = data.train_mask.clone()
        _atomic_pickle_dump(data, cache_path)
    return data


def _deduplicate_bidirected(data):
    """Remove one direction from each bidirected edge pair (keep u<v)."""
    import torch, copy
    import numpy as np
    from scipy.sparse import coo_matrix as _coo

    edge_index  = data.edge_index
    edge_weight = data.edge_weight

    src, dst = edge_index[0], edge_index[1]
    lo = torch.min(src, dst)
    hi = torch.max(src, dst)
    seen: dict = {}
    keep = []
    for i in range(edge_index.size(1)):
        key = (lo[i].item(), hi[i].item())
        if key not in seen:
            seen[key] = i
            keep.append(i)

    d = copy.copy(data)
    if len(keep) < edge_index.size(1):
        idx = torch.tensor(keep, dtype=torch.long)
        d.edge_index  = edge_index[:, idx]
        d.edge_weight = edge_weight[idx] if edge_weight is not None else None

    if hasattr(d, 'A'):
        ei = d.edge_index
        ew = d.edge_weight
        n  = int(ei.max().item()) + 1
        ew_np = ew.cpu().numpy() if ew is not None else np.ones(ei.size(1), dtype=np.float32)
        d.A = _coo(
            (ew_np, (ei[0].cpu().numpy(), ei[1].cpu().numpy())),
            shape=(n, n), dtype=np.float32,
        ).tocsr()
    return d


def get_link_split(data, dataset_name: str, run_id: int = None):
    """Return link split dict with train/val/test edges and labels. Caches result."""
    cache_dir = os.path.join(SPLIT_CACHE_DIR, 'link')
    os.makedirs(cache_dir, exist_ok=True)
    seed = link_split_seed(run_id) if run_id is not None else 50
    cache_name = (
        f'{dataset_name}_seed{seed}'
        f'_splits{LINK_SPLIT["splits"]}'
        f'_val{_fmt_ratio(LINK_SPLIT["prob_val"])}'
        f'_test{_fmt_ratio(LINK_SPLIT["prob_test"])}'
        '_task-sign.pkl'
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    with _cache_lock(cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        data_for_split = _deduplicate_bidirected(data)
        link_data = link_class_split(
            data_for_split,
            size=None,
            splits=LINK_SPLIT['splits'],
            prob_test=LINK_SPLIT['prob_test'],
            prob_val=LINK_SPLIT['prob_val'],
            task='sign',
            seed=seed,
            maintain_connect=False,
            ratio=1.0,
            device='cpu',
        )
        _atomic_pickle_dump(link_data, cache_path)
    return link_data
