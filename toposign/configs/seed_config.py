import random
import numpy as np
import torch

GENERATION_SEEDS = [0, 10, 20, 30, 40]
TRAINING_SEEDS   = [0, 10, 20, 30, 40]


def sdsbm_generation_seed(run_id: int) -> int:
    return GENERATION_SEEDS[run_id]


def link_split_seed(run_id: int) -> int:
    return GENERATION_SEEDS[run_id] + 50


def node_split_seed(run_id: int) -> int:
    return GENERATION_SEEDS[run_id]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
