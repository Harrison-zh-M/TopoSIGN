"""Input feature helpers for signed directed graph experiments."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from utils.general.in_out_degree import in_out_degree


def set_weighted_signed_input_features(data):
    """Use weighted 4D signed-directed degree features for GNN encoders."""
    num_nodes = data.x.size(0) if getattr(data, 'x', None) is not None else int(data.edge_index.max().item()) + 1
    if hasattr(data, 'separate_positive_negative'):
        data.separate_positive_negative()
    data.x = in_out_degree(
        data.edge_index,
        size=num_nodes,
        signed=True,
        edge_weight=data.edge_weight,
    )
    return data
