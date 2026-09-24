"""Generate dataset summary statistics and LaTeX table."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from statistics import fmean, pstdev

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric_signed_directed.data.signed.load_signed_real_data import load_signed_real_data
from toposign.configs.dataset_config import (
    REAL_DATASETS,
    RESULTS_DIR,
    SDSBM_DATA_DIR,
    SDSBM_SETTINGS,
)


def _load_sdsbm(setting_name: str, run_id: int):
    path = os.path.join(SDSBM_DATA_DIR, setting_name, f'run{run_id}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_real(dataset_name: str):
    return load_signed_real_data(dataset=dataset_name, root='data/')


def _num_nodes(data) -> int:
    if getattr(data, 'x', None) is not None:
        return int(data.x.size(0))
    return int(data.edge_index.max().item()) + 1


def _directed_edge_counts(data):
    edge_weight = data.edge_weight if getattr(data, 'edge_weight', None) is not None else torch.ones(data.edge_index.size(1))
    pos_edges = int((edge_weight > 0).sum().item())
    neg_edges = int((edge_weight < 0).sum().item())
    return pos_edges, neg_edges


def _sym_binary_sign_matrices(data, num_nodes: int):
    edge_index = data.edge_index.cpu().numpy()
    edge_weight = data.edge_weight.cpu().numpy() if getattr(data, 'edge_weight', None) is not None else np.ones(edge_index.shape[1])
    adj = sp.coo_matrix(
        (edge_weight, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()
    sym = sp.triu(adj + adj.T, k=1).tocoo()
    pos_mask = sym.data > 0
    neg_mask = sym.data < 0
    upper_pos = sp.coo_matrix(
        (np.ones(pos_mask.sum(), dtype=np.float64), (sym.row[pos_mask], sym.col[pos_mask])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    upper_neg = sp.coo_matrix(
        (np.ones(neg_mask.sum(), dtype=np.float64), (sym.row[neg_mask], sym.col[neg_mask])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    return upper_pos + upper_pos.T, upper_neg + upper_neg.T


def _unbalanced_triangles(data, num_nodes: int):
    pos_adj, neg_adj = _sym_binary_sign_matrices(data, num_nodes)
    adj = pos_adj + neg_adj
    total_triangles = float((adj @ adj @ adj).diagonal().sum()) / 6.0
    if total_triangles <= 0:
        return 0.0, 0.0
    one_negative = float((neg_adj @ pos_adj @ pos_adj).diagonal().sum()) / 2.0
    three_negative = float((neg_adj @ neg_adj @ neg_adj).diagonal().sum()) / 6.0
    unbalanced = one_negative + three_negative
    return unbalanced, 100.0 * unbalanced / total_triangles


def _stats_for_data(name: str, data) -> dict:
    n = _num_nodes(data)
    pos_edges, neg_edges = _directed_edge_counts(data)
    unbalanced, violation = _unbalanced_triangles(data, n)
    num_classes = int(data.y.max().item()) + 1 if getattr(data, 'y', None) is not None else 0
    avg_degree = (pos_edges + neg_edges) / n if n > 0 else 0.0
    return {
        'dataset': name,
        'num_nodes': float(n),
        'num_classes': float(num_classes),
        'positive_edges': float(pos_edges),
        'negative_edges': float(neg_edges),
        'avg_degree': float(avg_degree),
        'unbalanced_triangles': float(unbalanced),
        'violation_ratio_pct': float(violation),
    }


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(fmean(values)), float(pstdev(values))


def _aggregate_rows(instance_rows):
    grouped = defaultdict(list)
    for row in instance_rows:
        grouped[row['dataset']].append(row)

    rows = []
    metrics = [
        'num_nodes', 'num_classes', 'positive_edges', 'negative_edges',
        'avg_degree', 'unbalanced_triangles', 'violation_ratio_pct',
    ]
    for dataset in list(SDSBM_SETTINGS.keys()) + list(REAL_DATASETS):
        group_rows = grouped.get(dataset, [])
        if not group_rows:
            continue
        out = {'dataset': dataset, 'num_graphs': len(group_rows)}
        for metric in metrics:
            mean, std = _mean_std([row[metric] for row in group_rows])
            out[f'{metric}_mean'] = mean
            out[f'{metric}_std'] = std
        rows.append(out)
    return rows


def _format_number(mean: float, std: float, *, integer: bool = True) -> str:
    if integer:
        mean_text = f'{mean:,.0f}'
        std_text = f'{std:,.0f}'
    else:
        mean_text = f'{mean:.2f}'
        std_text = f'{std:.2f}'
    if std == 0 or not math.isfinite(std):
        return mean_text
    return rf'{mean_text} $\pm$ {std_text}'


def _latex_escape(value: str) -> str:
    return str(value).replace('_', r'\_').replace('&', r'\&')


def _latex_table(rows):
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Dataset summary statistics}',
        r'\label{tab:data-statistics}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lrrrrrrrr}',
        r'\hline',
        r'Dataset & Graphs & $n$ & $K$ & $|E^+|$ & $|E^-|$ & Avg. degree & $|\Delta_u|$ & $|\Delta_u|/|\Delta|$ (\%) \\',
        r'\hline',
    ]
    for row in rows:
        cells = [
            _latex_escape(row['dataset']),
            str(row['num_graphs']),
            _format_number(row['num_nodes_mean'], row['num_nodes_std']),
            _format_number(row['num_classes_mean'], row['num_classes_std']),
            _format_number(row['positive_edges_mean'], row['positive_edges_std']),
            _format_number(row['negative_edges_mean'], row['negative_edges_std']),
            _format_number(row['avg_degree_mean'], row['avg_degree_std'], integer=False),
            _format_number(row['unbalanced_triangles_mean'], row['unbalanced_triangles_std']),
            _format_number(row['violation_ratio_pct_mean'], row['violation_ratio_pct_std'], integer=False),
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([
        r'\hline',
        r'\end{tabular}%',
        r'}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


def generate_statistics(results_dir: str = None) -> None:
    results_dir = results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    instance_rows = []
    for setting in SDSBM_SETTINGS:
        for run_id in range(5):
            instance_rows.append(_stats_for_data(setting, _load_sdsbm(setting, run_id)))
    for dataset in REAL_DATASETS:
        instance_rows.append(_stats_for_data(dataset, _load_real(dataset)))

    rows = _aggregate_rows(instance_rows)

    json_path = os.path.join(results_dir, 'data_statistics.json')
    with open(json_path, 'w') as f:
        json.dump({'instances': instance_rows, 'summary': rows}, f, indent=2)

    csv_path = os.path.join(results_dir, 'data_statistics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    latex_dir = os.path.join(results_dir, 'latex')
    os.makedirs(latex_dir, exist_ok=True)
    latex_path = os.path.join(latex_dir, 'data_statistics.tex')
    with open(latex_path, 'w') as f:
        f.write(_latex_table(rows) + '\n')

    print(f'[data_statistics] wrote {json_path}')
    print(f'[data_statistics] wrote {csv_path}')
    print(f'[data_statistics] wrote {latex_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()
    generate_statistics(args.results_dir)
