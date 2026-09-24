import os
import math
import numpy as np

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SDSBM_DATA_DIR  = os.path.join(PROJECT_ROOT, 'toposign', 'data', 'sdsbm')
PI_TENSOR_DIR   = os.path.join(PROJECT_ROOT, 'toposign', 'data', 'pi_tensors')
POSITIVE_PI_TENSOR_DIR = os.path.join(PROJECT_ROOT, 'toposign', 'data', 'pi_tensors_positive')
SPLIT_CACHE_DIR = os.path.join(PROJECT_ROOT, 'toposign', 'data', 'splits')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'toposign', 'results')

SDSBM_SETTINGS = {
    'setting1': {'K': 3, 'p': 0.1, 'gamma': 0.25, 'eta': 0.25, 'N': 1000, 'size_ratio': 1.5, 'dataset_type': '3c'},
    'setting2': {'K': 4, 'p': 0.1, 'gamma': 0.25, 'eta': 0.25, 'N': 1000, 'size_ratio': 1.5, 'dataset_type': '4c'},
    'setting3': {'K': 4, 'p': 0.1, 'gamma': 0.10, 'eta': 0.25, 'N': 1000, 'size_ratio': 1.5, 'dataset_type': '4c'},
    'setting4': {'K': 3, 'p': 0.1, 'gamma': 0,    'eta': 0,    'N': 1000, 'size_ratio': 1.5, 'dataset_type': '3c'},
}

REAL_DATASETS = ['rainfall', 'sp1500']
SDSBM_RUN_IDS = list(range(5))
REAL_RUN_IDS  = list(range(5))
SDSBM_RUN_DATASETS = [
    f'{setting}_run{run_id}'
    for setting in SDSBM_SETTINGS
    for run_id in SDSBM_RUN_IDS
]
REAL_RUN_DATASETS = [
    f'{dataset}_run{run_id}'
    for dataset in REAL_DATASETS
    for run_id in REAL_RUN_IDS
]
ALL_EVAL_DATASETS = SDSBM_RUN_DATASETS + REAL_RUN_DATASETS
TABLE4_DATASETS = SDSBM_RUN_DATASETS
TABLE6_DATASETS = [f'setting{i}_run{j}' for i in range(1, 4) for j in range(5)]

TDA_CONFIG = {
    'k_hop': 2,
    'maxscale': 10.0,
    'topk': 80,
    'pixel_size': 1.0,
    'is_landmarks': False,
}

LINK_SPLIT = {
    'splits': 10,
    'prob_val': 0.1,
    'prob_test': 0.1,
}

NODE_SPLIT = {
    'seed_ratio': 0.1,
    'val_ratio': 0.1,
}

TRAIN_CONFIG = {
    'epochs': 1000,
    'lr': 1e-2,
    'weight_decay': 5e-4,
    'dropout': 0.5,
    'hidden': 32,
    'early_stopping': 400,
}

PRETRAIN_CONFIG = {
    'epochs': 1000,
    'early_stopping': 400,
    'lr': 1e-2,
}

FINETUNE_CONFIG = {
    'epochs': 100,
    'early_stopping': 40,
}

TABLE2_CONFIG = {
    'epochs': 1000,
    'early_stopping': 400,
}

HYPERPARAMETER_GRID = {
    'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
}


def pi_tensor_filename(dataset_name: str, tda_type: str = 'signed',
                       pixel_size: float = None) -> str:
    k = TDA_CONFIG['k_hop']
    ms = TDA_CONFIG['maxscale']
    px = TDA_CONFIG['pixel_size'] if pixel_size is None else pixel_size
    px_str = str(px).replace('.', 'p')
    ms_str = str(ms).replace('.', 'p')
    return f'{dataset_name}_k{k}_max{ms_str}_px{px_str}.pt'


def pi_tensor_dir(tda_type: str = 'signed', override_dir: str = None) -> str:
    if override_dir:
        return override_dir
    env_name = (
        'SDG_DEBUG_POSITIVE_PI_TENSOR_DIR'
        if tda_type == 'positive'
        else 'SDG_DEBUG_PI_TENSOR_DIR'
    )
    env_dir = os.environ.get(env_name)
    if env_dir:
        return env_dir
    return POSITIVE_PI_TENSOR_DIR if tda_type == 'positive' else PI_TENSOR_DIR


def resolve_pi_tensor_path(dataset_name: str, tda_type: str = 'signed',
                           override_dir: str = None,
                           pixel_size: float = None) -> str:
    directory = pi_tensor_dir(tda_type, override_dir)
    candidates = [
        os.path.join(directory, pi_tensor_filename(dataset_name, tda_type, pixel_size=pixel_size)),
    ]
    if pixel_size is None or pixel_size != 0.1:
        candidates.append(
            os.path.join(directory, pi_tensor_filename(dataset_name, tda_type, pixel_size=0.1))
        )
    candidates.append(os.path.join(directory, f'{dataset_name}.pt'))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _infer_pixel_size(flat_dim: int, maxscale: float) -> float:
    side = int(round(math.sqrt(flat_dim)))
    return maxscale / side


def split_real_run_dataset(dataset_name: str):
    if '_run' not in dataset_name:
        return dataset_name, None
    base, run_part = dataset_name.rsplit('_run', 1)
    if base in REAL_DATASETS:
        return base, int(run_part)
    return dataset_name, None


def split_sdsbm_run_dataset(dataset_name: str):
    if '_run' not in dataset_name:
        return dataset_name, None
    base, run_part = dataset_name.rsplit('_run', 1)
    if base in SDSBM_SETTINGS:
        return base, int(run_part)
    return dataset_name, None


def get_F_matrix(dataset_type: str, gamma: float) -> np.ndarray:
    if dataset_type == '3c':
        return np.array([
            [0.5,        gamma,    -gamma],
            [1 - gamma,  0.5,      -0.5],
            [-1 + gamma, -0.5,      0.5],
        ])
    elif dataset_type == '4c':
        return np.array([
            [0.5,        gamma,    -gamma,     gamma],
            [1 - gamma,  0.5,      -gamma,    -0.5],
            [-1 + gamma, -1 + gamma, 0.5,     -1 + gamma],
            [-1 + gamma, -0.5,     -gamma,     0.5],
        ])
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
