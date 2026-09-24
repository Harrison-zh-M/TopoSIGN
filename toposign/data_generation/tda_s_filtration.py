import numpy as np
import gudhi as gd
import networkx as nx
from tqdm import tqdm

import torch
from persim import PersistenceImager

from typing import Any, List, Tuple





def edge_index_to_dense_matrix(edge_index: torch.LongTensor, num_node: int):
    """Convert edge_index to dense adjacency matrix"""
    node_idx = torch.LongTensor([i for i in range(num_node)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    sp_adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size((num_node, num_node)))
    return sp_adj.to_dense().numpy()


def edge_index_weight_to_dense_matrix(edge_index: torch.LongTensor, edge_weight: torch.Tensor, num_node: int):
    """Convert signed weighted edge_index to dense adjacency matrix."""
    if edge_index is None or edge_index.numel() == 0:
        return np.zeros((num_node, num_node), dtype=np.float32)

    edge_index_np = edge_index.cpu().numpy()
    edge_weight_np = edge_weight.cpu().numpy() if isinstance(edge_weight, torch.Tensor) else np.array(edge_weight, dtype=np.float32)
    adj = np.zeros((num_node, num_node), dtype=np.float32)

    for idx in range(edge_index_np.shape[1]):
        i = int(edge_index_np[0, idx])
        j = int(edge_index_np[1, idx])
        adj[i, j] += float(edge_weight_np[idx])
    return adj


def signed_adjacency_from_data(data: Any) -> np.ndarray:
    """Return a dense signed adjacency matrix for SignedData or SDSBM-like graphs."""
    num_node = data.x.size(0)

    if hasattr(data, 'edge_index_p') and hasattr(data, 'edge_index_n'):
        pos_weight = data.edge_weight_p if hasattr(data, 'edge_weight_p') else torch.ones(data.edge_index_p.size(1), dtype=torch.float32)
        neg_weight = data.edge_weight_n if hasattr(data, 'edge_weight_n') else torch.ones(data.edge_index_n.size(1), dtype=torch.float32)
        pos_adj = edge_index_weight_to_dense_matrix(data.edge_index_p, pos_weight, num_node)
        neg_adj = edge_index_weight_to_dense_matrix(data.edge_index_n, neg_weight, num_node)
        return pos_adj - neg_adj

    if hasattr(data, 'edge_index') and hasattr(data, 'edge_weight'):
        return edge_index_weight_to_dense_matrix(data.edge_index, data.edge_weight, num_node)

    if hasattr(data, 'edge_index'):
        return edge_index_to_dense_matrix(data.edge_index, num_node)

    raise ValueError('Cannot build adjacency matrix from data object.')

def compute_tda_from_signed_data(data: Any,
                                 k_hop: int = 2,
                                 maxscale: float = 10.0,
                                 is_landmarks: bool = False,
                                 topk: int = 80,
                                 pixel_size: float = 0.1,
                                 include_empty_images: bool = False) -> Tuple[List[np.ndarray], torch.Tensor]:
    """Compute Dowker persistence diagrams and persistent images from signed graph data."""
    adj_array = signed_adjacency_from_data(data).astype(np.float32)

    if is_landmarks:
        degree = np.sum(np.abs(adj_array), axis=0)
        thresh = np.percentile(degree, topk)
        landmarks = np.where(degree >= thresh)[0]
    else:
        landmarks = None

    k_hop_subgraphs = (k_th_order_subgraph_landmarks(adj_mat=adj_array, k=k_hop, landmarks=landmarks)
                       if is_landmarks else k_th_order_subgraph(adj_mat=adj_array, k=k_hop))

    dowker_filtration_dgms = []
    print('\nBegin Dowker Filtration based TDA...')
    for i in tqdm(range(len(k_hop_subgraphs))):
        if is_landmarks and i not in landmarks:
            dowker_filtration_dgms.append(np.array([]))
            continue

        k_hop_subgraph = k_hop_subgraphs[i]
        degree_vectors = compute_signed_degree_vectors(k_hop_subgraph)
        distance_matrix = compute_distance_matrix(degree_vectors)
        dowker_dgm = dowker_filtration_dgm(degree_vectors, k_hop_subgraph, distance_matrix, maxscale=maxscale)
        dowker_filtration_dgms.append(dowker_dgm if dowker_dgm.size else np.array([]))

    print('Done Dowker Filtration based TDA!')
    print(f'Total diagrams generated: {len(dowker_filtration_dgms)}')

    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=(0, maxscale), pers_range=(0, maxscale))
    zero_image = None
    if include_empty_images:
        zero_image = np.zeros_like(pimgr.transform(np.array([[0.0, maxscale]], dtype=np.float32)).flatten())
    persistent_images = []
    for dgm in dowker_filtration_dgms:
        if len(dgm) > 0:
            persistent_images.append(pimgr.transform(dgm).flatten())
        elif include_empty_images:
            persistent_images.append(zero_image)

    pi_tensor = torch.FloatTensor(np.array(persistent_images))
    print(f'Persistent images shape: {pi_tensor.shape}')
    return pi_tensor

    """ for dgm in dowker_filtration_dgms:
        if len(dgm) > 0:
            persistent_images.append(pimgr.transform(dgm))

    pi_tensor = torch.FloatTensor(np.array(persistent_images))
    print(f'Persistent images shape: {pi_tensor.shape}')
    return pi_tensor"""
    


def k_th_order_subgraph(adj_mat, k):
    """Extract k-hop subgraphs for all nodes"""
    print("\nBegin k-th subgraph extraction...")
    output = list()
    G = nx.from_numpy_array(adj_mat)
    for i in tqdm(range(adj_mat.shape[0])):
        v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
        tmp_subgraph = adj_mat[np.ix_(v_labels, v_labels)]
        output.append(tmp_subgraph)
    print("Done k-th subgraph extraction!")
    return output


def k_th_order_subgraph_landmarks(adj_mat, k, landmarks):
    """Extract k-hop subgraphs centered at landmark nodes"""
    print("\nBegin k-th subgraph extraction...")
    output = list()
    G = nx.from_numpy_array(adj_mat)
    for i in tqdm(range(adj_mat.shape[0])):
        if i in landmarks:
            v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
            tmp_subgraph = adj_mat[np.ix_(v_labels, v_labels)]
            output.append(tmp_subgraph)
        else:
            output.append([])
    print("Done k-th subgraph extraction!")
    return output


def compute_signed_degree_vectors(subgraph):
    """
    Compute signed degree vectors for each node in the subgraph
    
    For each node u in the subgraph:
    - d_plus = sum of weights of positive edges incident to u
    - d_minus = sum of absolute values of weights of negative edges incident to u
    
    Returns:
        degree_vectors: (num_nodes, 2) array where each row is [d_plus, d_minus]
    """
    num_nodes = subgraph.shape[0]
    degree_vectors = np.zeros((num_nodes, 2), dtype=np.float32)
    
    for u in range(num_nodes):
        # Get all edges incident to node u (both directions for undirected graph)
        edges_u = subgraph[u, :]
        
        # d_plus: sum of positive edge weights
        d_plus = np.sum(edges_u[edges_u > 0])
        
        # d_minus: sum of absolute values of negative edge weights
        d_minus = np.sum(np.abs(edges_u[edges_u < 0]))
        
        degree_vectors[u, 0] = d_plus
        degree_vectors[u, 1] = d_minus
    
    return degree_vectors


def compute_distance_matrix(degree_vectors):
    """
    Compute pairwise Euclidean distances in signed degree space
    
    For nodes i and j:
    δ_±(i,j) = ||s_i - s_j||_2 = sqrt((d_plus_i - d_plus_j)^2 + (d_minus_i - d_minus_j)^2)
    
    Args:
        degree_vectors: (num_nodes, 2) array of [d_plus, d_minus] vectors
        
    Returns:
        distance_matrix: (num_nodes, num_nodes) symmetric matrix of pairwise distances
    """
    num_nodes = degree_vectors.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Euclidean distance between degree vectors
            diff = degree_vectors[i] - degree_vectors[j]
            distance_matrix[i, j] = np.linalg.norm(diff)
    
    return distance_matrix


def dowker_filtration_dgm(degree_vectors, subgraph, distance_matrix, maxscale=None):
    """
    Build Dowker filtration complex and compute persistence diagrams
    
    For each simplex σ, its filtration value is:
    filtration(σ) = min_w (max_{ℓ∈σ} δ_±(ℓ, w))
    
    where:
    - σ is a simplex (node or edge)
    - w is a witness node (any node in the complex)
    - δ_± is the Euclidean distance in signed degree space
    - For a node [i]: max(δ_±(i, w)) = δ_±(i, w)
    - For an edge [i,j]: max(δ_±(i, w), δ_±(j, w))
    
    Args:
        degree_vectors: (num_nodes, 2) array of signed degree vectors
        distance_matrix: (num_nodes, num_nodes) pairwise distance matrix
        maxscale: optional maximum scale for capping filtration values
        
    Returns:
        persistence_diagram: (num_features, 2) array of (birth, death) pairs
    """
    num_nodes = degree_vectors.shape[0]
    
    # Initialize SimplexTree with 0-simplices (nodes)
    stb = gd.SimplexTree()
    
    # Add all nodes (0-simplices) with filtration value 0.0
    for node in range(num_nodes):
        stb.insert([node], filtration=0.0)
    
    # 仅针对原图中存在的边计算过滤值
    xs, ys = np.where(np.abs(subgraph) > 0)

    # Compute all filtration values for edges (1-simplices)
    edge_filtrations = []
    for i, j in zip(xs, ys):
            # For edge [i,j], compute filtration = min_w (max(δ_±(i,w), δ_±(j,w)))
            min_max_dist = float('inf')
            min_max_dist = np.min(np.maximum(distance_matrix[i, :], distance_matrix[j, :]))
            
            # Apply maxscale capping if specified
            if maxscale is not None and min_max_dist > maxscale:
                min_max_dist = maxscale
            
            edge_filtrations.append((i, j, min_max_dist))
    
    # Add edges in increasing order of filtration value
    for i, j, filt_val in edge_filtrations:
        stb.insert([i, j], filtration=filt_val)
    
    # Ensure filtration is non-decreasing
    stb.make_filtration_non_decreasing()
    
    # Compute persistence
    dgm = stb.persistence()
    
    # Extract persistence pairs and format as (birth, death)
    pd = []
    for dim, (birth, death) in dgm:
        if np.isfinite(death):
            pd.append([birth, death])
        else:
            # Handle infinite death (persistence to infinity)
            if maxscale is not None:
                pd.append([birth, maxscale])
            else:
                # Use a large value for infinite persistence
                pd.append([birth, birth + 1000.0])
    
    if len(pd) == 0:
        return np.array([])
    else:
        return np.array(pd, dtype=np.float32)
