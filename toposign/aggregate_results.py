"""Read JSONL result files and generate console plus LaTeX tables."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import glob
from collections import defaultdict
from statistics import fmean, pstdev
import math
import re
from typing import Callable, Dict, Iterable, List

from toposign.configs.dataset_config import (
    ALL_EVAL_DATASETS, REAL_DATASETS, RESULTS_DIR, SDSBM_SETTINGS, TABLE6_DATASETS
)


TABLE_CONFIGS = {
    'table1_pretrain_finetune': {
        'title': 'Table 1: Pre-training and Prompt Learning',
        'label': 'tab:main_compare_gfms',
        'caption': (
            r'Pre-training and prompt learning performance comparison in terms of'
            r' downstream node clustering ARI. We report the mean ARI among five'
            r' runs plus/minus one standard deviation. The best method is marked in'
            r' \textbf{bold} while the second best is marked with \underline{underline}.'
        ),
        'our_methods': ['TopoMSGNN', 'TopoSSSNET'],
        'our_method_labels': {
            'TopoMSGNN': r'TopoMSGNN (Ours)',
            'TopoSSSNET': r'TopoSSSNET (Ours)',
        },
        'full_datasets': True,
        'methods': [
            'GPPT', 'Gprompt', 'GPF', 'All-in-one', 'SAMGPT', 'TopoDIG',
            'TopoMSGNN', 'TopoSSSNET',
        ],
    },
    'table2_node_clustering': {
        'title': 'Table 2: Node Clustering Baselines',
        'label': 'tab:main_results_other_signed_GNNs',
        'caption': (
            r'Semi-supervised node clustering performance (ARI) on real-world and'
            r' synthetic signed datasets. We compare our approach against state-of-the-art'
            r' signed GNNs. We report the mean ARI among five runs plus/minus one standard'
            r' deviation. The best method is marked in \textbf{bold} while the second best'
            r' is marked with \underline{underline}.'
        ),
        'our_methods': ['DSGC+Topo', 'SigMaNet+Topo', 'MSGNN+Topo', 'SSSNET+Topo'],
        'our_method_labels': {
            'DSGC+Topo':    r'DSGC+Topo (Ours)',
            'SigMaNet+Topo': r'SigMaNet+Topo (Ours)',
            'MSGNN+Topo':   r'MSGNN+Topo (Ours)',
            'SSSNET+Topo':  r'SSSNET+Topo (Ours)',
        },
        'full_datasets': True,
        'methods': [
            'SSSNET', 'SigMaNet', 'MSGNN', 'DSGC',
            'DSGC+Topo', 'SigMaNet+Topo', 'MSGNN+Topo', 'SSSNET+Topo',
        ],
    },
    'table3_filtration_ablation': {
        'title': 'Table 3: Filtration Ablation',
        'label': 'tab:ablation_filtration',
        'caption': (
            r'Filtration ablation: effect of different topological filtration strategies'
            r' on node clustering ARI for TopoMSGNN (top) and TopoSSSNET (bottom).'
            r' We report the mean ARI among five runs plus/minus'
            r' one standard deviation. The best method is marked in \textbf{bold} while'
            r' the second best is marked with \underline{underline}.'
        ),
        'full_datasets': True,
        'methods': [
            'MSGNN_only', 'Topo_only', 'MSGNN+posTopo', 'TopoMSGNN',
            'SSSNET_only', 'SSSNET+posTopo', 'TopoSSSNET',
        ],
    },
    'table4_pretrain_task_ablation': {
        'title': 'Table 4: Pre-training Task Ablation',
        'label': 'tab:ablation_pretrain',
        'caption': (
            r'Pre-training task ablation: effect of different pre-training objectives on'
            r' downstream node clustering ARI after prompt learning for TopoMSGNN and'
            r' TopoSSSNET. We report the mean ARI among five runs'
            r' plus/minus one standard deviation. The best method is marked in'
            r' \textbf{bold} while the second best is marked with \underline{underline}.'
        ),
        'methods': [
            'TopoMSGNN_pretrain_SP',
            'TopoMSGNN_pretrain_SP+DP',
            'TopoMSGNN_pretrain_SP+3C',
            'TopoMSGNN_pretrain_SP+4C',
            'TopoMSGNN_pretrain_SP+5C',
            'TopoSSSNET_pretrain_SP',
            'TopoSSSNET_pretrain_SP+DP',
            'TopoSSSNET_pretrain_SP+3C',
            'TopoSSSNET_pretrain_SP+4C',
            'TopoSSSNET_pretrain_SP+5C',
        ],
    },
    'table5_prompt_comparison': {
        'title': 'Table 5: Prompt Method Comparison',
        'label': 'tab:ablation_prompt_comparison',
        'caption': (
            r'Prompt method comparison: performance of TopoMSGNN and TopoSSSNET combined'
            r' with different graph prompting strategies. We report the mean ARI among'
            r' five runs plus/minus one standard deviation. The best method is marked in'
            r' \textbf{bold} while the second best is marked with \underline{underline}.'
        ),
        'methods': [
            'TopoMSGNN+GPPT', 'TopoMSGNN+Gprompt', 'TopoMSGNN+GPF', 'TopoMSGNN+All-in-one',
            'TopoSSSNET+GPPT', 'TopoSSSNET+Gprompt', 'TopoSSSNET+GPF', 'TopoSSSNET+All-in-one',
        ],
    },
    'table6_backbone_ablation': {
        'title': 'Table 6: Backbone Ablation',
        'label': 'tab:ablation_other_signed_backbones',
        'caption': (
            r'Backbone ablation: performance of our topological augmentation applied to'
            r' different signed GNN backbones. We report the mean ARI among five runs'
            r' plus/minus one standard deviation. The best method is marked in'
            r' \textbf{bold} while the second best is marked with \underline{underline}.'
        ),
        'full_datasets': False,   # only SDSBM-1, 2, 3
        'methods': ['TopoSigMaNet', 'TopoDSGC', 'TopoMSGNN', 'TopoSSSNET'],
    },
}


RUN_DATASET_ORDER = list(ALL_EVAL_DATASETS)
SETTING_DATASET_ORDER = ['SDSBM-1', 'SDSBM-2', 'SDSBM-3', 'SDSBM-4', 'Rainfall', 'SP1500']

# Tables 4, 5, and 6 share identical TopoMSGNN/TopoSSSNET runs with Table 1
# (same pre-training, same GPPT prompt, same signed filtration). Rather than
# re-running those experiments, we copy Table 1 records with remapped method
# names so that the comparison rows are always consistent with Table 1.
_CROSS_LOAD = {
    'table4_pretrain_task_ablation': {
        'source': 'table1_pretrain_finetune',
        'renames': {
            'TopoMSGNN':  'TopoMSGNN_pretrain_SP',
            'TopoSSSNET': 'TopoSSSNET_pretrain_SP',
        },
    },
    'table5_prompt_comparison': {
        'source': 'table1_pretrain_finetune',
        'renames': {
            'TopoMSGNN':  'TopoMSGNN+GPPT',
            'TopoSSSNET': 'TopoSSSNET+GPPT',
        },
    },
    'table6_backbone_ablation': {
        'source': 'table1_pretrain_finetune',
        'renames': {
            'TopoMSGNN':  'TopoMSGNN',
            'TopoSSSNET': 'TopoSSSNET',
        },
    },
}

METHOD_LABEL_ALIASES = {
    'TPL_only': 'Topo_only',
    'TDA_only': 'Topo_only',
    'MSGC_only': 'MSGNN_only',
    'SGE_only': 'MSGNN_only',
    'MSGC+posTopo': 'MSGNN+posTopo',
    'SGE+posTopo': 'MSGNN+posTopo',
    'MSGC+posTDA': 'MSGNN+posTopo',
    'MSGC+positiveTDA': 'MSGNN+posTopo',
    'MSGC+positiveTopo': 'MSGNN+posTopo',
    # TopoSIGN → TopoMSGNN (old JSONL entries)
    'TopoSIGN': 'TopoMSGNN',
    'TopoSIGN+GPPT': 'TopoMSGNN+GPPT',
    'TopoSIGN+Gprompt': 'TopoMSGNN+Gprompt',
    'TopoSIGN+GPF': 'TopoMSGNN+GPF',
    'TopoSIGN+All-in-one': 'TopoMSGNN+All-in-one',
    'TopoSIGN_pretrain_SP': 'TopoMSGNN_pretrain_SP',
    'TopoSIGN_pretrain_SP+DP': 'TopoMSGNN_pretrain_SP+DP',
    'TopoSIGN_pretrain_SP+3C': 'TopoMSGNN_pretrain_SP+3C',
    'TopoSIGN_pretrain_SP+4C': 'TopoMSGNN_pretrain_SP+4C',
    'TopoSIGN_pretrain_SP+5C': 'TopoMSGNN_pretrain_SP+5C',
    # Table 6 backbone names
    'MSGNN+Topo+GPPT': 'TopoMSGNN',
    'SSSNET+Topo+GPPT': 'TopoSSSNET',
    'SigMaNet+Topo+GPPT': 'TopoSigMaNet',
    'DSGC+Topo+GPPT': 'TopoDSGC',
    # legacy TDA aliases
    'SSSNET+TDA+GPPT': 'TopoSSSNET',
    'SigMaNet+TDA+GPPT': 'TopoSigMaNet',
    'MSGNN+TDA+GPPT': 'TopoMSGNN',
    'DSGC+TDA+GPPT': 'TopoDSGC',
    'TopoDIG+TDA+GPPT': 'TopoDIG+Topo+GPPT',
    'TopoSIGN+TDA+All-in-one': 'TopoMSGNN+All-in-one',
    'TopoSIGN+TDA+GPPT': 'TopoMSGNN+GPPT',
}


def _normalize_method_label(method: str) -> str:
    return METHOD_LABEL_ALIASES.get(method, method)


def _normalize_record(record: dict) -> dict:
    record = dict(record)
    if 'method' in record:
        record['method'] = _normalize_method_label(record['method'])
    return record


def _debug_sized_curve_file(path: str) -> bool:
    try:
        with open(path, 'r') as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if record.get('debug', False):
        return True
    epochs_run = int(record.get('epochs_run', 0) or 0)
    train_loss = record.get('train_loss', [])
    return epochs_run <= 2 or len(train_loss) <= 2


def _debug_only_record(results_dir: str, table: str, record: dict) -> bool:
    method = record.get('method')
    dataset = record.get('dataset')
    if not method or not dataset:
        return False
    pattern = os.path.join(
        results_dir, 'curves', f'{table}_{method}_{dataset}_seed*.json'
    )
    curve_paths = glob.glob(pattern)
    return bool(curve_paths) and all(_debug_sized_curve_file(path) for path in curve_paths)


def load_jsonl(path: str, no_debug_filter: bool = False) -> list:
    records = []
    results_dir = os.path.dirname(path)
    table = os.path.splitext(os.path.basename(path))[0]
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if no_debug_filter or (
                    not record.get('debug', False)
                    and not _debug_only_record(results_dir, table, record)
                ):
                    records.append(_normalize_record(record))
    return records


def _dedupe_latest(records: Iterable[dict]) -> List[dict]:
    latest = {}
    for index, rec in enumerate(records):
        key = (rec.get('method'), rec.get('dataset'))
        latest[key] = (index, rec)
    return [rec for _, rec in sorted(latest.values(), key=lambda item: item[0])]


def _ordered_values(values: Iterable[str], preferred: Iterable[str]) -> List[str]:
    values = list(values)
    seen = []
    for value in preferred:
        if value in values and value not in seen:
            seen.append(value)
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def print_table(records: list, title: str) -> None:
    records = _dedupe_latest(records)
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # Group by method and dataset
    grouped = defaultdict(dict)
    methods  = []
    datasets = []
    for rec in records:
        method  = rec['method']
        dataset = rec['dataset']
        if method not in methods:
            methods.append(method)
        if dataset not in datasets:
            datasets.append(dataset)
        grouped[method][dataset] = (rec['mean_ari'], rec['std_ari'])

    # Header
    col_w = 22
    ds_w  = 18
    header = f"{'Method':<{col_w}}" + "".join(f"{d:<{ds_w}}" for d in datasets)
    print(header)
    print('-' * len(header))

    for method in methods:
        row = f"{method:<{col_w}}"
        for dataset in datasets:
            if dataset in grouped[method]:
                mean, std = grouped[method][dataset]
                cell = f"{mean:.3f} +/- {std:.3f}"
            else:
                cell = "--"
            row += f"{cell:<{ds_w}}"
        print(row)

    print()


def _latex_escape(value: str) -> str:
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(char, char) for char in str(value))


def _latex_label(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9:._-]+', '-', str(value))


_DATASET_DISPLAY = {
    'setting1': 'SDSBM-1',
    'setting2': 'SDSBM-2',
    'setting3': 'SDSBM-3',
    'setting4': 'SDSBM-4',
    'rainfall': 'Rainfall',
    'sp1500':   'SP1500',
}


def _display_dataset(dataset: str, dataset_mode: str) -> str:
    if dataset_mode == 'settings':
        match = re.match(r'^(setting[0-9]+|rainfall|sp1500)(?:_run\d+)?$', dataset)
        if match:
            return _DATASET_DISPLAY.get(match.group(1), match.group(1))
    return dataset


def _stats_from_records(records: List[dict], metric: str = 'ari'):
    if not records:
        return None
    if metric == 'runtime':
        values = []
        for rec in records:
            values.extend(float(v) for v in rec.get('runtime_values', []))
        if not values:
            return None
        mean = fmean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
    else:
        if len(records) == 1:
            mean = float(records[0]['mean_ari'])
            std = float(records[0]['std_ari'])
        else:
            means = [float(rec['mean_ari']) for rec in records]
            mean = fmean(means)
            std = pstdev(means) if len(means) > 1 else 0.0
    if not math.isfinite(mean) or not math.isfinite(std):
        return None
    return mean, std


def _cell_from_stats(stats, metric: str = 'ari', bold: bool = False,
                     underline: bool = False) -> str:
    if stats is None:
        return '--'
    mean, std = stats
    if metric == 'runtime':
        body = rf'{mean:.0f} \pm {std:.0f}'
    else:
        body = rf'{mean:.3f} \pm {std:.3f}'
    if bold:
        return rf'$\mathbf{{{body}}}$'
    if underline:
        return rf'\underline{{${body}$}}'
    return rf'${body}$'


def _cell_from_records(records: List[dict], metric: str = 'ari',
                       bold: bool = False, underline: bool = False) -> str:
    return _cell_from_stats(_stats_from_records(records, metric), metric=metric,
                            bold=bold, underline=underline)


def _latex_table(
    records: List[dict],
    *,
    title: str,
    label: str,
    caption: str = None,
    method_order: List[str],
    our_method: str = None,
    our_method_label: str = None,
    our_methods: List[str] = None,
    our_method_labels: Dict[str, str] = None,
    dataset_mode: str = 'settings',
    metric: str = 'ari',
    full_datasets: bool = False,
    resizebox: bool = True,
) -> str:
    records = _dedupe_latest(records)
    if not records:
        return f'% No records found for {title}'

    buckets = defaultdict(lambda: defaultdict(list))
    methods_present = []
    datasets_present = []
    for rec in records:
        method = rec['method']
        dataset = _display_dataset(rec['dataset'], dataset_mode)
        if method not in methods_present:
            methods_present.append(method)
        if dataset not in datasets_present:
            datasets_present.append(dataset)
        buckets[method][dataset].append(rec)

    dataset_order = SETTING_DATASET_ORDER if dataset_mode == 'settings' else RUN_DATASET_ORDER
    if full_datasets and dataset_mode == 'settings':
        datasets = list(SETTING_DATASET_ORDER)
    else:
        datasets = _ordered_values(datasets_present, dataset_order)
    methods = _ordered_values(methods_present, method_order)

    alignment = 'l' + ('c' * len(datasets))
    header = (
        r'\textbf{Method} & '
        + ' & '.join(rf'\textbf{{{_latex_escape(ds)}}}' for ds in datasets)
        + r' \\'
    )

    best_by_dataset = {}
    second_best_by_dataset = {}
    if metric in ('ari', 'runtime'):
        for dataset in datasets:
            candidates = []
            for method in methods:
                stats = _stats_from_records(buckets[method][dataset], metric=metric)
                if stats is not None:
                    candidates.append((method, stats[0]))
            if candidates:
                # For ARI: higher is better; for runtime: lower is better
                if metric == 'runtime':
                    best_val = min(v for _, v in candidates)
                    cmp = lambda v: abs(v - best_val) <= 1e-9
                    below = [(m, v) for m, v in candidates if abs(v - best_val) > 1e-9]
                    second_val = min(v for _, v in below) if below else None
                else:
                    best_val = max(v for _, v in candidates)
                    cmp = lambda v: abs(v - best_val) <= 1e-12
                    below = [(m, v) for m, v in candidates if abs(v - best_val) > 1e-12]
                    second_val = max(v for _, v in below) if below else None
                best_by_dataset[dataset] = {m for m, v in candidates if cmp(v)}
                if second_val is not None:
                    if metric == 'runtime':
                        second_best_by_dataset[dataset] = {
                            m for m, v in below if abs(v - second_val) <= 1e-9
                        }
                    else:
                        second_best_by_dataset[dataset] = {
                            m for m, v in below if abs(v - second_val) <= 1e-12
                        }

    caption_text = caption if caption is not None else _latex_escape(title)
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        rf'\caption{{{caption_text}}}',
        rf'\label{{{_latex_label(label)}}}',
    ]
    if resizebox:
        lines.append(r'\resizebox{\textwidth}{!}{%')
    lines += [
        rf'\begin{{tabular}}{{{alignment}}}',
        r'\toprule',
        header,
        r'\midrule',
    ]
    # Build ours set and label lookup
    _our_set: set = set()
    if our_methods:
        _our_set = set(our_methods)
    elif our_method:
        _our_set = {our_method}
    _our_labels: Dict[str, str] = {}
    if our_method_labels:
        _our_labels = our_method_labels
    elif our_method_label and our_method:
        _our_labels = {our_method: our_method_label}

    for method in methods:
        is_our = method in _our_set
        if is_our:
            lines.append(r'\midrule')
        cells = []
        for dataset in datasets:
            stats = _stats_from_records(buckets[method][dataset], metric=metric)
            bold = method in best_by_dataset.get(dataset, set())
            underline = (not bold) and method in second_best_by_dataset.get(dataset, set())
            cells.append(_cell_from_stats(stats, metric=metric, bold=bold, underline=underline))
        if is_our:
            display = _our_labels.get(method, method)
            row = rf'\textbf{{{_latex_escape(display)}}} & ' + ' & '.join(cells) + r' \\'
        else:
            row = _latex_escape(method) + ' & ' + ' & '.join(cells) + r' \\'
        lines.append(row)
    if resizebox:
        lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    else:
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    return '\n'.join(lines)


def _runtime_records_for_table(results_dir: str, table_name: str,
                               records: List[dict]) -> List[dict]:
    curves_dir = os.path.join(results_dir, 'curves')
    if not os.path.isdir(curves_dir):
        return []

    wanted = {(rec.get('method'), rec.get('dataset')) for rec in records}
    by_key_seed = defaultdict(lambda: defaultdict(float))
    for path in glob.glob(os.path.join(curves_dir, '*.json')):
        try:
            with open(path, 'r') as f:
                curve = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if curve.get('table') != table_name:
            continue
        key = (_normalize_method_label(curve.get('method')), curve.get('dataset'))
        if key not in wanted:
            continue
        if 'elapsed_sec' not in curve or 'seed' not in curve:
            continue
        by_key_seed[key][int(curve['seed'])] += float(curve['elapsed_sec'])

    runtime_records = []
    for method, dataset in sorted(wanted):
        values = list(by_key_seed.get((method, dataset), {}).values())
        if not values:
            continue
        runtime_records.append({
            'method': method,
            'dataset': dataset,
            'runtime_values': values,
        })
    return runtime_records


def _table_latex(key: str, records: List[dict], dataset_mode: str = 'settings') -> str:
    cfg = TABLE_CONFIGS[key]
    return _latex_table(
        records,
        title=cfg['title'],
        label=cfg['label'],
        caption=cfg.get('caption'),
        method_order=cfg['methods'],
        our_method=cfg.get('our_method'),
        our_method_label=cfg.get('our_method_label'),
        our_methods=cfg.get('our_methods'),
        our_method_labels=cfg.get('our_method_labels'),
        full_datasets=cfg.get('full_datasets', False),
        dataset_mode=dataset_mode,
    )


def generate_table1_latex(records: List[dict], dataset_mode: str = 'settings') -> str:
    return _table_latex('table1_pretrain_finetune', records, dataset_mode)


def generate_table2_latex(records: List[dict], dataset_mode: str = 'settings') -> str:
    return _table_latex('table2_node_clustering', records, dataset_mode)


def generate_table3_latex(records: List[dict], dataset_mode: str = 'settings') -> str:
    return _table_latex('table3_filtration_ablation', records, dataset_mode)


def generate_table4_latex(records: List[dict], dataset_mode: str = 'settings') -> str:
    return _table_latex('table4_pretrain_task_ablation', records, dataset_mode)


def generate_table5_latex(records: List[dict], dataset_mode: str = 'settings') -> str:
    return _table_latex('table5_prompt_comparison', records, dataset_mode)


def generate_table6_latex(records: List[dict], dataset_mode: str = 'settings') -> str:
    # Filter to TABLE6_DATASETS (settings 1-3 only)
    t6_set = set(TABLE6_DATASETS)
    records = [rec for rec in records if rec.get('dataset') in t6_set]
    return _table_latex('table6_backbone_ablation', records, dataset_mode)


TABLE_GENERATORS: Dict[str, Callable[[List[dict], str], str]] = {
    'table1_pretrain_finetune':    generate_table1_latex,
    'table2_node_clustering':      generate_table2_latex,
    'table3_filtration_ablation':  generate_table3_latex,
    'table4_pretrain_task_ablation': generate_table4_latex,
    'table5_prompt_comparison':    generate_table5_latex,
    'table6_backbone_ablation':    generate_table6_latex,
}


def generate_dataset_stats_latex(results_dir: str) -> str:
    """Generate a dataset statistics table including SDSBM-4 computed from data files."""
    import pickle
    data_root = os.path.join(os.path.dirname(results_dir), 'data', 'sdsbm')

    hardcoded = {
        'SDSBM-1': {'n': 1000, 'K': 3, 'pos_mean': 25426, 'pos_std': 109,
                    'neg_mean': 24565, 'neg_std': 76, 'deg': 49.99, 'deg_std': 0.12},
        'SDSBM-2': {'n': 1000, 'K': 4, 'pos_mean': 21949, 'pos_std': 101,
                    'neg_mean': 27958, 'neg_std': 128, 'deg': 49.91, 'deg_std': 0.13},
        'SDSBM-3': {'n': 1000, 'K': 4, 'pos_mean': 21514, 'pos_std': 129,
                    'neg_mean': 28433, 'neg_std': 163, 'deg': 49.95, 'deg_std': 0.20},
        'Rainfall': {'n': 306, 'K': 6, 'pos_mean': 64408, 'pos_std': None,
                     'neg_mean': 29228, 'neg_std': None, 'deg': 306.00, 'deg_std': None},
        'SP1500':  {'n': 1193, 'K': 10, 'pos_mean': 1069319, 'pos_std': None,
                    'neg_mean': 353930, 'neg_std': None, 'deg': 1193.00, 'deg_std': None},
    }

    # Compute SDSBM-4 dynamically if data exists
    setting4_dir = os.path.join(data_root, 'setting4')
    sdsbm4 = None
    if os.path.isdir(setting4_dir):
        pos_counts, neg_counts, degs = [], [], []
        n_val, k_val = None, None
        for run_id in range(5):
            pkl_path = os.path.join(setting4_dir, f'run{run_id}.pkl')
            if not os.path.exists(pkl_path):
                continue
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                import torch
                ew = data.edge_weight
                n_pos = int((ew > 0).sum().item())
                n_neg = int((ew < 0).sum().item())
                n_nodes = data.num_nodes
                k = int(data.y.max().item()) + 1
                pos_counts.append(n_pos)
                neg_counts.append(n_neg)
                degs.append((n_pos + n_neg) / n_nodes)
                n_val = n_nodes
                k_val = k
            except Exception:
                continue
        if pos_counts:
            sdsbm4 = {
                'n': n_val, 'K': k_val,
                'pos_mean': round(fmean(pos_counts)),
                'pos_std': round(pstdev(pos_counts)) if len(pos_counts) > 1 else 0,
                'neg_mean': round(fmean(neg_counts)),
                'neg_std': round(pstdev(neg_counts)) if len(neg_counts) > 1 else 0,
                'deg': fmean(degs),
                'deg_std': pstdev(degs) if len(degs) > 1 else 0.0,
            }

    def _fmt_edge(mean, std):
        s = f'{mean:,}'
        if std is not None and std > 0:
            s += rf' $\pm$ {std:,}'
        return s

    def _fmt_deg(deg, std):
        if std is not None and std > 0:
            return rf'{deg:.2f} $\pm$ {std:.2f}'
        return f'{deg:.2f}'

    rows_order = ['SDSBM-1', 'SDSBM-2', 'SDSBM-3', 'SDSBM-4', 'Rainfall', 'SP1500']
    data_rows = dict(hardcoded)
    if sdsbm4:
        data_rows['SDSBM-4'] = sdsbm4

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Dataset summary statistics, where SDSBM results are averaged over'
        r' five random seeds and we also report one standard deviation.}',
        r'\label{tab:data-statistics}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lrrrrrrrr}',
        r'\toprule',
        r'Dataset & $n$ & $K$ & $|\mathcal{E}^+|$ & $|\mathcal{E}^-|$ & Avg.\ degree \\',
        r'\midrule',
    ]
    for ds in rows_order:
        if ds not in data_rows:
            continue
        d = data_rows[ds]
        edge_pos = _fmt_edge(d['pos_mean'], d.get('pos_std'))
        edge_neg = _fmt_edge(d['neg_mean'], d.get('neg_std'))
        deg = _fmt_deg(d['deg'], d.get('deg_std'))
        lines.append(rf'{ds} & {d["n"]:,} & {d["K"]} & {edge_pos} & {edge_neg} & {deg} \\')
    lines.extend([r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}'])
    return '\n'.join(lines)


def _write_latex(table_name: str, latex_code: str, latex_dir: str) -> str:
    os.makedirs(latex_dir, exist_ok=True)
    path = os.path.join(latex_dir, f'{table_name}.tex')
    with open(path, 'w') as f:
        f.write(latex_code + '\n')
    return path


def main(
    results_dir: str = None,
    *,
    latex_dir: str = None,
    latex_dataset_mode: str = 'settings',
    latex_only: bool = False,
    no_latex: bool = False,
    no_debug_filter: bool = False,
) -> None:
    results_dir = results_dir or RESULTS_DIR
    latex_dir = latex_dir or os.path.join(results_dir, 'latex')
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    jsonl_files = sorted(glob.glob(os.path.join(results_dir, '*.jsonl')))
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return

    # Pre-load table1 records once for cross-table injection.
    _t1_records_cache: dict = {}

    def _cross_load(table_name: str, records: list) -> list:
        cfg = _CROSS_LOAD.get(table_name)
        if cfg is None:
            return records
        source = cfg['source']
        if source not in _t1_records_cache:
            src_path = os.path.join(results_dir, f'{source}.jsonl')
            _t1_records_cache[source] = (
                load_jsonl(src_path, no_debug_filter=no_debug_filter)
                if os.path.exists(src_path) else []
            )
        renames = cfg['renames']
        extra = []
        for rec in _t1_records_cache[source]:
            m = rec.get('method')
            if m in renames:
                rec2 = dict(rec)
                rec2['method'] = renames[m]
                extra.append(rec2)
        # Append after table-specific records so _dedupe_latest keeps Table 1
        # values (they are appended last and therefore "most recent").
        return records + extra

    for path in jsonl_files:
        table_name = os.path.basename(path).replace('.jsonl', '')
        records = load_jsonl(path, no_debug_filter=no_debug_filter)
        records = _cross_load(table_name, records)
        if not records:
            continue
        cfg = TABLE_CONFIGS.get(table_name, {})
        title = cfg.get('title', table_name.replace('_', ' ').title())
        if not latex_only:
            print_table(records, title)

        if no_latex:
            continue
        generator = TABLE_GENERATORS.get(table_name)
        if generator is None:
            continue
        latex_code = generator(records, latex_dataset_mode)
        out_path = _write_latex(table_name, latex_code, latex_dir)
        print(f"\n[latex] {out_path}")
        print(latex_code)

        runtime_records = _runtime_records_for_table(results_dir, table_name, records)
        if runtime_records:
            base_label = cfg.get('label', table_name)
            if base_label.startswith('tab:'):
                runtime_label = 'tab:runtime_' + base_label[4:]
            else:
                runtime_label = base_label + '-runtime'
            runtime_caption = (
                rf'Average runtime (seconds) per run for each method on each seed,'
                rf' plus/minus one standard deviation, for result'
                rf' Table~\ref{{{_latex_label(base_label)}}}.'
                rf' The shortest time is marked with \textbf{{bold}} while the second'
                rf' shortest time is marked with \underline{{underline}}.'
            )
            runtime_latex = _latex_table(
                runtime_records,
                title=f"{title} Runtime",
                label=runtime_label,
                caption=runtime_caption,
                method_order=cfg.get('methods', []),
                full_datasets=cfg.get('full_datasets', False),
                dataset_mode=latex_dataset_mode,
                metric='runtime',
                resizebox=False,
            )
            runtime_path = _write_latex(f'{table_name}_runtime', runtime_latex, latex_dir)
            print(f"\n[latex-runtime] {runtime_path}")
            print(runtime_latex)

    if not no_latex:
        # Dataset statistics table
        stats_latex = generate_dataset_stats_latex(results_dir)
        stats_path = _write_latex('dataset_statistics', stats_latex, latex_dir)
        print(f"\n[latex] {stats_path}")
        print(stats_latex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--latex_dir', type=str, default=None)
    parser.add_argument('--latex_dataset_mode', choices=['settings', 'runs'],
                        default='settings',
                        help='settings collapses settingN_runM columns into settingN.')
    parser.add_argument('--latex_only', action='store_true')
    parser.add_argument('--no_latex', action='store_true')
    parser.add_argument('--no_debug_filter', action='store_true',
                        help='Include results with epochs_run<=2 (e.g. smoke-test runs)')
    pargs = parser.parse_args()
    main(
        pargs.results_dir,
        latex_dir=pargs.latex_dir,
        latex_dataset_mode=pargs.latex_dataset_mode,
        latex_only=pargs.latex_only,
        no_latex=pargs.no_latex,
        no_debug_filter=pargs.no_debug_filter,
    )
