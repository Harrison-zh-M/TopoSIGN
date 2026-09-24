"""Training runners for link sign pre-training and node clustering fine-tuning.

Both runners record per-epoch loss curves and wall-clock time for every
(model, dataset, seed) triple and persist them to:

  {results_dir}/curves/{table}_{method}_{dataset}_seed{seed}.json

Schema of each curve file:
  {
    "table":        "table2_node_clustering",
    "method":       "SSSNET",
    "dataset":      "setting1_run0",
    "seed":         0,
    "epochs_run":   128,
    "elapsed_sec":  14.7,
    "train_loss":   [0.693, 0.651, ...],
    "val_ari":      [0.00, 0.03, ...],
    "best_val_ari": 0.41,
    "test_ari":     0.39
  }
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import fcntl
import math
from typing import Tuple, List, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from toposign.configs.seed_config import seed_everything, node_split_seed
from toposign.configs.dataset_config import (
    SDSBM_SETTINGS, REAL_DATASETS,
    ALL_EVAL_DATASETS, SDSBM_DATA_DIR, PI_TENSOR_DIR, POSITIVE_PI_TENSOR_DIR, RESULTS_DIR,
    PRETRAIN_CONFIG, HYPERPARAMETER_GRID,
    split_real_run_dataset, split_sdsbm_run_dataset, resolve_pi_tensor_path,
)


def training_config_from_args(args, extra: dict = None) -> dict:
    """Compact, JSON-safe hyperparameter provenance for curve files."""
    keys = (
        'hidden', 'topo_hidden', 'sigmanet_hidden', 'num_layers', 'K', 'q',
        'dropout', 'normalization', 'lr', 'gppt_lr', 'weight_decay',
        'epochs', 'early_stopping', 'pretrain_batch_size', 'pixel_size', 'tda_type',
        'use_structural', 'sd_input_features', 'weighted_input_features',
        'sd_input_feat', 'weighted_input_feat', 'hparam_search',
    )
    config = {}
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        elif isinstance(value, (list, tuple)):
            config[key] = list(value)
        else:
            config[key] = str(value)
    if extra:
        config.update(extra)
    return config


def lr_to_tag(lr: float) -> str:
    return f'{float(lr):.8g}'.replace('-', 'm').replace('.', 'p')


def apply_finetune_lr(args, lr: float) -> None:
    """Set every fine-tuning LR field a prompt method might actually read."""
    args.lr = float(lr)
    if hasattr(args, 'gppt_lr'):
        args.gppt_lr = float(lr)


class RunResult(NamedTuple):
    test_ari:    float
    best_val_ari: float
    elapsed_sec: float
    epochs_run:  int
    train_loss:  List[float]
    val_ari_curve: List[float]


class LinkSignPretrainer:
    """Pre-trains encoder with BCE loss on link sign prediction."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device

    def train_epoch(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        train_edges: torch.LongTensor,
        train_labels: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(real, imag, edge_index, edge_weight, pi)
        logits = self.model.edge_decoder(z, train_edges)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        query_edges: torch.LongTensor,
        query_labels: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
    ) -> float:
        self.model.eval()
        z = self.model.encode(real, imag, edge_index, edge_weight, pi)
        logits = self.model.edge_decoder(z, query_edges)
        preds = (logits > 0).long()
        return (preds == query_labels).float().mean().item()


class NodeClusteringRunner:
    """Multi-seed training and ARI evaluation for node clustering."""

    def __init__(
        self,
        model_cls,
        model_kwargs: dict,
        data,
        args,
        pi: Optional[torch.Tensor] = None,
        *,
        table: str = '',
        method: str = '',
        dataset: str = '',
        results_dir: str = '',
    ):
        self.model_cls    = model_cls
        self.model_kwargs = model_kwargs
        self.data         = data
        self.args         = args
        self.pi           = pi
        self.table        = table
        self.method       = method
        self.dataset      = dataset
        self.results_dir  = results_dir or getattr(args, 'results_dir', '')
        self.device       = torch.device(
            f'cuda:{getattr(args, "device_id", 0)}'
            if torch.cuda.is_available() and not getattr(args, 'cpu', False)
            else 'cpu'
        )

    def _run_one(self, seed: int) -> RunResult:
        seed_everything(seed)
        t_start = time.perf_counter()

        data        = self.data
        pi          = self.pi.to(self.device) if self.pi is not None else None
        real        = data.x.to(self.device)
        imag        = data.x.to(self.device)
        edge_index  = data.edge_index.to(self.device)
        edge_weight = data.edge_weight.to(self.device) if data.edge_weight is not None else None
        y           = data.y.to(self.device)

        train_mask = data.train_mask[:, 0].to(self.device)
        val_mask   = data.val_mask[:, 0].to(self.device)
        test_mask  = data.test_mask[:, 0].to(self.device)

        model = self.model_cls(**self.model_kwargs).to(self.device)

        num_nodes = int(edge_index.max().item()) + 1
        if hasattr(model, 'set_laplacian'):
            model.set_laplacian(edge_index, edge_weight, num_nodes, self.device)
        elif hasattr(model, '_inner') and hasattr(model._inner, 'set_laplacian'):
            model._inner.set_laplacian(edge_index, edge_weight, num_nodes, self.device)
        elif hasattr(model, 'model') and hasattr(model.model, 'set_laplacian'):
            model.model.set_laplacian(edge_index, edge_weight, num_nodes, self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        best_val_ari  = -1.0
        best_test_ari = 0.0
        patience      = 0
        early_stopping = getattr(self.args, 'early_stopping', 400)

        train_loss_curve: List[float] = []
        val_ari_curve:    List[float] = []

        for epoch in range(self.args.epochs):
            model.train()
            z, log_prob, pred, prob = _forward(model, real, imag, edge_index, edge_weight, pi, data)
            loss = finite_nll_loss(log_prob[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_curve.append(round(loss.item(), 6))

            model.eval()
            with torch.no_grad():
                _, _, pred_val, _ = _forward(model, real, imag, edge_index, edge_weight, pi, data)

            val_ari = adjusted_rand_score(
                y[val_mask].cpu().numpy(),
                pred_val[val_mask].cpu().numpy(),
            )
            val_ari_curve.append(round(float(val_ari), 6))

            if val_ari > best_val_ari:
                best_val_ari  = val_ari
                best_test_ari = adjusted_rand_score(
                    y[test_mask].cpu().numpy(),
                    pred_val[test_mask].cpu().numpy(),
                )
                patience = 0
            else:
                patience += 1

            if patience >= early_stopping:
                break

        elapsed = time.perf_counter() - t_start

        return RunResult(
            test_ari    = float(best_test_ari),
            best_val_ari = float(best_val_ari),
            elapsed_sec = round(elapsed, 3),
            epochs_run  = len(train_loss_curve),
            train_loss  = train_loss_curve,
            val_ari_curve = val_ari_curve,
        )

    def run(self) -> Tuple[float, float]:
        """Run over all training seeds. Saves curve files. Returns (mean_ari, std_ari)."""
        import numpy as np
        aris: List[float] = []
        self.last_seeds: List[int] = []
        self.last_ari_values: List[float] = []

        for seed in self.args.seeds:
            cached = None
            if not getattr(self.args, 'force', False):
                cached = load_curve_record(
                    self.results_dir, self.table, self.method, self.dataset, seed,
                    args=self.args,
                )
            if cached is not None:
                result = RunResult(
                    test_ari=float(cached['test_ari']),
                    best_val_ari=float(cached.get('best_val_ari', cached['test_ari'])),
                    elapsed_sec=float(cached.get('elapsed_sec', 0.0)),
                    epochs_run=int(cached.get('epochs_run', 0)),
                    train_loss=list(cached.get('train_loss', [])),
                    val_ari_curve=list(cached.get('val_ari_curve', [])),
                )
                print(f'  seed={seed:2d}  ARI={result.test_ari:.3f}  restored from curve checkpoint')
            elif getattr(self.args, 'hparam_search', False):
                result, best_lr = self._hparam_tune_seed(seed)
                print(f'  seed={seed:2d}  selected_lr={best_lr:g}  ARI={result.test_ari:.3f}')
            else:
                result = self._run_one(seed)
                self._save_curve(seed, result)
                print(f'  seed={seed:2d}  ARI={result.test_ari:.3f}  epochs={result.epochs_run}  time={result.elapsed_sec:.1f}s')
            aris.append(result.test_ari)
            self.last_seeds.append(int(seed))
            self.last_ari_values.append(float(result.test_ari))

        return float(np.mean(aris)), float(np.std(aris))

    def _save_curve(self, seed: int, result: RunResult, config_extra: dict = None) -> None:
        if not self.results_dir:
            return
        os.makedirs(os.path.join(self.results_dir, 'curves'), exist_ok=True)
        path  = curve_path(self.results_dir, self.table, self.method, self.dataset, seed)
        record = {
            'table':         self.table,
            'method':        self.method,
            'dataset':       self.dataset,
            'seed':          seed,
            'epochs_run':    result.epochs_run,
            'elapsed_sec':   result.elapsed_sec,
            'best_val_ari':  round(result.best_val_ari, 6),
            'test_ari':      round(result.test_ari, 6),
            'train_loss':    result.train_loss,
            'val_ari_curve': result.val_ari_curve,
            'debug':         bool(getattr(self.args, 'debug', False)),
            'config':        training_config_from_args(self.args, extra=config_extra),
        }
        with open(path, 'w') as f:
            json.dump(record, f)

    def _hparam_tune_seed(self, seed: int) -> Tuple[RunResult, float]:
        original_lr = getattr(self.args, 'lr', None)
        original_gppt_lr = getattr(self.args, 'gppt_lr', None)
        best_lr = None
        best_result = None

        for lr_candidate in (getattr(self.args, 'lr_candidates', None) or HYPERPARAMETER_GRID['lr']):
            apply_finetune_lr(self.args, lr_candidate)
            result = self._run_one(seed)
            lr_tag = lr_to_tag(lr_candidate)
            save_curve_file(
                results_dir=self.results_dir,
                table=self.table,
                method=self.method,
                dataset=self.dataset,
                seed=seed,
                epochs_run=result.epochs_run,
                elapsed_sec=result.elapsed_sec,
                train_loss=result.train_loss,
                val_ari_curve=result.val_ari_curve,
                best_val_ari=result.best_val_ari,
                test_ari=result.test_ari,
                phase=f'hparam_lr{lr_tag}',
                debug=bool(getattr(self.args, 'debug', False)),
                config=training_config_from_args(
                    self.args,
                    extra={
                        'hparam_search_scope': 'per_seed_lr',
                        'hparam_phase': True,
                        'selection_seed': seed,
                        'candidate_lr': float(lr_candidate),
                        'candidate_finetune_lr': float(lr_candidate),
                        'candidate_gppt_lr': float(getattr(self.args, 'gppt_lr', lr_candidate)),
                        'candidate_lrs': list(HYPERPARAMETER_GRID['lr']),
                        'tuned_lr_fields': ['lr', 'gppt_lr'],
                    },
                ),
            )
            print(f'    [hparam seed={seed}] lr={lr_candidate:g}  val={result.best_val_ari:.3f}  test={result.test_ari:.3f}')
            if best_result is None or result.best_val_ari > best_result.best_val_ari:
                best_result = result
                best_lr = float(lr_candidate)

        apply_finetune_lr(self.args, best_lr)
        self._save_curve(
            seed,
            best_result,
            config_extra={
                'hparam_search_scope': 'per_seed_lr',
                'selected_lr': best_lr,
                'selected_finetune_lr': best_lr,
                'selected_gppt_lr': float(getattr(self.args, 'gppt_lr', best_lr)),
                'selected_seed': seed,
                'candidate_lrs': list(HYPERPARAMETER_GRID['lr']),
                'tuned_lr_fields': ['lr', 'gppt_lr'],
            },
        )
        if original_lr is not None:
            self.args.lr = original_lr
        if original_gppt_lr is not None:
            self.args.gppt_lr = original_gppt_lr
        return best_result, best_lr


def ari_summary_metadata(seeds, aris) -> dict:
    return {
        'seeds': [int(seed) for seed in seeds],
        'ari_values': [round(float(ari), 6) for ari in aris],
        'std_scope': 'test_ari_across_training_seeds',
        'ari_metric_scope': 'one_ari_over_all_test_nodes_per_seed',
    }


def training_seeds_for_run(args, run_id) -> List[int]:
    if run_id is not None:
        return [int(node_split_seed(run_id))]
    requested = getattr(args, '_requested_seeds', getattr(args, 'seeds', []))
    return [int(seed) for seed in requested]


def save_result(
    results_dir: str,
    table: str,
    method: str,
    dataset: str,
    mean_ari: float,
    std_ari: float,
    extra: dict = None,
) -> None:
    import time as _t
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'{table}.jsonl')
    row  = {
        'method':    method,
        'dataset':   dataset,
        'mean_ari':  round(mean_ari, 3),
        'std_ari':   round(std_ari,  3),
        'timestamp': _t.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra:
        row.update(extra)
    append_jsonl(path, row)
    print(f'  saved → {path}  ({mean_ari:.3f} ± {std_ari:.3f})')


def result_exists(results_dir: str, table: str, method: str, dataset: str, args=None) -> bool:
    path = os.path.join(results_dir, f'{table}.jsonl')
    if not os.path.exists(path):
        return False
    with open(path, 'r') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    row.get('method') == method
                    and row.get('dataset') == dataset
                    and not (row.get('debug') and not getattr(args, 'debug', False))
                ):
                    expected_seeds = [int(seed) for seed in getattr(args, 'seeds', [])]
                    if expected_seeds:
                        row_seeds = row.get('seeds')
                        if row_seeds is None:
                            continue
                        if [int(seed) for seed in row_seeds] != expected_seeds:
                            continue
                    return True
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return False


def _is_debug_sized_curve(record: dict) -> bool:
    if bool(record.get('debug', False)):
        return True
    epochs_run = int(record.get('epochs_run', 0) or 0)
    train_loss = record.get('train_loss', [])
    return epochs_run <= 2 or len(train_loss) <= 2


def _has_nonfinite_values(record: dict) -> bool:
    keys = ('test_ari', 'best_val_ari', 'elapsed_sec')
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return True
    for key in ('train_loss', 'val_ari_curve'):
        values = record.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return True
    return False


def _curve_record_is_complete(record: dict, args=None) -> bool:
    if 'test_ari' not in record:
        return False
    if _has_nonfinite_values(record):
        return False
    if getattr(args, 'hparam_search', False):
        config = record.get('config', {})
        if config.get('hparam_search_scope') != 'per_seed_lr':
            return False
    if getattr(args, 'debug', False):
        return True
    return not _is_debug_sized_curve(record)


def _final_curve_phase(table: str):
    if table in {
        'table1_pretrain_finetune',
        'table4_pretrain_task_ablation',
        'table5_prompt_comparison',
        'table6_backbone_ablation',
    }:
        return 'finetune'
    return None


def _curve_resume_status(args, results_dir: str, table: str, method: str, dataset: str) -> str:
    seeds = list(getattr(args, 'seeds', []) or [])
    if not seeds:
        return 'unknown'
    found_any = False
    phase = _final_curve_phase(table)
    for seed in seeds:
        path = curve_path(results_dir, table, method, dataset, seed, phase=phase)
        if not os.path.exists(path):
            if found_any:
                return 'invalid'
            return 'unknown'
        found_any = True
        try:
            with open(path, 'r') as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 'invalid'
        if not _curve_record_is_complete(record, args=args):
            return 'invalid'
    return 'complete' if found_any else 'unknown'


def should_skip_result(args, results_dir: str, table: str, method: str, dataset: str) -> bool:
    if getattr(args, 'force', False):
        return False
    if result_exists(results_dir, table, method, dataset, args=args):
        if not getattr(args, 'debug', False):
            status = _curve_resume_status(args, results_dir, table, method, dataset)
            if status == 'invalid':
                print(f'  rerun existing aggregate with incomplete/debug curves: {table} / {method} / {dataset}')
                return False
        print(f'  skip existing result: {table} / {method} / {dataset}')
        return True
    return False


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row) + '\n')
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def curve_path(
    results_dir: str,
    table: str,
    method: str,
    dataset: str,
    seed: int,
    phase: str = None,
) -> str:
    curves_dir = os.path.join(results_dir, 'curves')
    if phase:
        tag = f'{table}_{method}_{dataset}_seed{seed}_{phase}'
    else:
        tag = f'{table}_{method}_{dataset}_seed{seed}'
    return os.path.join(curves_dir, f'{tag}.json')


def load_curve_record(
    results_dir: str,
    table: str,
    method: str,
    dataset: str,
    seed: int,
    phase: str = None,
    args=None,
) -> Optional[dict]:
    path = curve_path(results_dir, table, method, dataset, seed, phase=phase)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not _curve_record_is_complete(record, args=args):
        return None
    return record


def save_curve_file(
    results_dir: str,
    table: str,
    method: str,
    dataset: str,
    seed: int,
    epochs_run: int,
    elapsed_sec: float,
    train_loss: List[float],
    val_ari_curve: List[float],
    best_val_ari: float,
    test_ari: float,
    phase: str = 'finetune',
    debug: bool = False,
    config: dict = None,
) -> None:
    """Save per-run loss curve and timing to {results_dir}/curves/{tag}.json."""
    if not results_dir:
        return
    curves_dir = os.path.join(results_dir, 'curves')
    os.makedirs(curves_dir, exist_ok=True)
    path = curve_path(results_dir, table, method, dataset, seed, phase=phase)
    record = {
        'table':         table,
        'method':        method,
        'dataset':       dataset,
        'seed':          seed,
        'phase':         phase,
        'epochs_run':    epochs_run,
        'elapsed_sec':   round(elapsed_sec, 3),
        'best_val_ari':  round(best_val_ari, 6),
        'test_ari':      round(test_ari, 6),
        'train_loss':    train_loss,
        'val_ari_curve': val_ari_curve,
        'debug':         bool(debug),
    }
    if config is not None:
        record['config'] = config
    with open(path, 'w') as f:
        json.dump(record, f)


def run_per_seed_lr_search(
    *,
    pretrained_model: nn.Module,
    data,
    pi,
    args,
    device: torch.device,
    run_finetune_fn,
    results_dir: str,
    table: str,
    method: str,
    dataset: str,
    seed: int,
) -> Tuple[float, float]:
    """Tune fine-tuning LR for one seed, saving every candidate and the winner."""
    from copy import deepcopy

    best = None
    for lr_candidate in (getattr(args, 'lr_candidates', None) or HYPERPARAMETER_GRID['lr']):
        seed_everything(seed)
        candidate_args = deepcopy(args)
        apply_finetune_lr(candidate_args, lr_candidate)
        candidate_model = deepcopy(pretrained_model)

        test_ari, val_ari, train_loss, val_ari_curve, elapsed = run_finetune_fn(
            candidate_model, data, pi, candidate_args, device
        )
        lr_tag = lr_to_tag(lr_candidate)
        save_curve_file(
            results_dir=results_dir,
            table=table,
            method=method,
            dataset=dataset,
            seed=seed,
            epochs_run=len(train_loss),
            elapsed_sec=elapsed,
            train_loss=train_loss,
            val_ari_curve=val_ari_curve,
            best_val_ari=val_ari,
            test_ari=test_ari,
            phase=f'hparam_lr{lr_tag}',
            debug=getattr(args, 'debug', False),
            config=training_config_from_args(
                candidate_args,
                extra={
                    'hparam_search_scope': 'per_seed_lr',
                    'hparam_phase': True,
                    'selection_seed': seed,
                    'candidate_lr': float(lr_candidate),
                    'candidate_finetune_lr': float(lr_candidate),
                    'candidate_gppt_lr': float(getattr(candidate_args, 'gppt_lr', lr_candidate)),
                    'candidate_lrs': list(HYPERPARAMETER_GRID['lr']),
                    'tuned_lr_fields': ['lr', 'gppt_lr'],
                },
            ),
        )
        print(f'    [hparam seed={seed}] lr={lr_candidate:g}  val={val_ari:.3f}  test={test_ari:.3f}')
        if best is None or val_ari > best['val_ari']:
            best = {
                'lr': float(lr_candidate),
                'test_ari': float(test_ari),
                'val_ari': float(val_ari),
                'train_loss': train_loss,
                'val_ari_curve': val_ari_curve,
                'elapsed': elapsed,
                'args': candidate_args,
            }

    save_curve_file(
        results_dir=results_dir,
        table=table,
        method=method,
        dataset=dataset,
        seed=seed,
        epochs_run=len(best['train_loss']),
        elapsed_sec=best['elapsed'],
        train_loss=best['train_loss'],
        val_ari_curve=best['val_ari_curve'],
        best_val_ari=best['val_ari'],
        test_ari=best['test_ari'],
        phase='finetune',
        debug=getattr(args, 'debug', False),
        config=training_config_from_args(
            best['args'],
            extra={
                'hparam_search_scope': 'per_seed_lr',
                'selected_lr': best['lr'],
                'selected_finetune_lr': best['lr'],
                'selected_gppt_lr': float(getattr(best['args'], 'gppt_lr', best['lr'])),
                'selected_seed': seed,
                'candidate_lrs': list(HYPERPARAMETER_GRID['lr']),
                'tuned_lr_fields': ['lr', 'gppt_lr'],
            },
        ),
    )
    return best['test_ari'], best['lr']


def run_prompt_finetune(model, data, pi, args, device: torch.device):
    """Fine-tune a prompt/head module and return curves without saving files."""
    optimizer = torch.optim.Adam(
        prompt_tuning_parameters(model),
        lr=prompt_tuning_lr(model, args),
        weight_decay=args.weight_decay,
    )
    t_start = time.perf_counter()

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

    best_val_ari = -1.0
    best_test_ari = 0.0
    patience = 0
    early_stopping = getattr(args, 'early_stopping', 400)
    train_loss_curve = []
    val_ari_curve = []

    for epoch in range(args.epochs):
        model.train()
        _data = _make_data_obj(real, imag, edge_index, edge_weight, data)
        _data.y = y
        _, log_prob, _, _ = sanitize_model_output(*model(_data, pi_dev))
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
            best_val_ari = val_ari
            best_test_ari = float(adjusted_rand_score(
                y[test_mask].cpu().numpy(), pred_val[test_mask].cpu().numpy()
            ))
            patience = 0
        else:
            patience += 1
        if getattr(args, 'debug', False) and epoch >= 1:
            break
        if patience >= early_stopping:
            break

    return best_test_ari, best_val_ari, train_loss_curve, val_ari_curve, time.perf_counter() - t_start


def prompt_tuning_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Freeze encoder weights and return prompt/head parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = False

    trainable = []
    prompt_module_names = (
        'prompt', 'answering', 'classifier', 'node_prompt',
        'task_head', 'PG', 'downstreamPrompt',
    )
    for name in prompt_module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        if isinstance(module, nn.Parameter):
            params = [module]
        else:
            params = list(module.parameters())
        for param in params:
            param.requires_grad = True
            trainable.append(param)

    if not trainable:
        for param in model.parameters():
            param.requires_grad = True
            trainable.append(param)
    return trainable


def prompt_tuning_lr(model: nn.Module, args) -> float:
    if model.__class__.__name__ == 'SignedGPPT':
        return float(getattr(args, 'gppt_lr', 2e-3))
    return float(getattr(args, 'lr', 1e-2))


def sanitize_model_output(z, log_prob, pred=None, prob=None):
    if log_prob is None:
        return z, log_prob, pred, prob
    finite_scores = torch.nan_to_num(log_prob, nan=0.0, neginf=-1.0e9, posinf=1.0e9)
    prob = F.softmax(finite_scores, dim=1)
    log_prob = torch.log(prob.clamp_min(1.0e-12))
    pred = prob.argmax(dim=1)
    return z, log_prob, pred, prob


def finite_nll_loss(log_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    safe_log_prob = torch.nan_to_num(log_prob, nan=-27.631021, neginf=-27.631021, posinf=0.0)
    return F.nll_loss(safe_log_prob, target)


def finite_nll_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    safe_logits = torch.nan_to_num(logits, nan=0.0, neginf=-1.0e9, posinf=1.0e9)
    return finite_nll_loss(F.log_softmax(safe_logits, dim=1), target)


def ensure_edge_index_2xe(edges: torch.Tensor) -> torch.LongTensor:
    if edges.dim() != 2:
        raise ValueError(f'Expected a 2D edge tensor, got shape {tuple(edges.shape)}')
    if edges.size(0) == 2:
        return edges.long().contiguous()
    if edges.size(1) == 2:
        return edges.t().long().contiguous()
    raise ValueError(f'Cannot infer edge orientation from shape {tuple(edges.shape)}')


def observed_sign_edges(data, device: torch.device = None) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Return observed graph edges with binary positive-sign labels."""
    edge_index = ensure_edge_index_2xe(data.edge_index)
    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is None:
        labels = torch.ones(edge_index.size(1), dtype=torch.float32)
    else:
        labels = (edge_weight > 0).float()
    if device is not None:
        edge_index = edge_index.to(device)
        labels = labels.to(device)
    return edge_index, labels


def sample_link_batch(
    edges: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Sample a class-balanced pretraining link batch (equal positive/negative)."""
    edges = ensure_edge_index_2xe(edges)
    labels = labels.view(-1)
    num_edges = edges.size(1)
    if labels.numel() != num_edges:
        raise ValueError(
            f'Edge/label size mismatch: edges={tuple(edges.shape)} labels={tuple(labels.shape)}'
        )
    if batch_size <= 0 or num_edges <= batch_size:
        return edges, labels

    pos_idx = (labels > 0.5).nonzero(as_tuple=True)[0]
    neg_idx = (labels <= 0.5).nonzero(as_tuple=True)[0]

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        perm = torch.randperm(num_edges, device=edges.device)[:batch_size]
        return edges[:, perm], labels[perm]

    half = batch_size // 2
    pos_sel = pos_idx[torch.randperm(pos_idx.numel(), device=edges.device)[:half]]
    neg_sel = neg_idx[torch.randperm(neg_idx.numel(), device=edges.device)[:half]]

    perm = torch.cat([pos_sel, neg_sel])
    perm = perm[torch.randperm(perm.size(0), device=edges.device)]
    return edges[:, perm], labels[perm]


def _forward(model, real, imag, edge_index, edge_weight, pi, original_data):
    """Dispatch forward() regardless of model type (TopoSIGN vs baseline vs GFM)."""
    if hasattr(model, 'tda_type'):
        return sanitize_model_output(*model(real, imag, edge_index, edge_weight, pi))
    else:
        _data = _make_data_obj(real, imag, edge_index, edge_weight, original_data)
        return sanitize_model_output(*model(_data, pi))


def _make_data_obj(real, imag, edge_index, edge_weight, original_data):
    class _Data:
        pass
    d              = _Data()
    d.x            = real
    d.edge_index   = edge_index
    d.edge_weight  = edge_weight
    d.y            = original_data.y.to(real.device) if hasattr(original_data, 'y') else None
    for attr in ('edge_index_p', 'edge_weight_p', 'edge_index_n', 'edge_weight_n'):
        val = getattr(original_data, attr, None)
        setattr(d, attr, val.to(real.device) if val is not None else None)
    return d
