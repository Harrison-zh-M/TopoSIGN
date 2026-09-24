from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_relu import complex_relu_layer
from .MSConv import MSConv

class MSGNN_link_prediction(nn.Module):
    r"""The MSGNN model for link prediction.

    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 1.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=1, label_dim:int=2, \
        activation:bool=True, trainable_q:bool=False, layer:int=2, dropout:float=0.5, normalization:str='sym', cached: bool=False, absolute_degree: bool=True):
        super(MSGNN_link_prediction, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached, absolute_degree=absolute_degree))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden*4, label_dim)
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        query_edges: torch.LongTensor, edge_weight: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real[query_edges[:,0]], real[query_edges[:,1]], imag[query_edges[:,0]], imag[query_edges[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        self.z = x.clone()
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class MSGNN_node_classification(nn.Module):
    r"""The MSGNN model for node classification.

    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 1.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`False`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} \odot \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} \odot \exp(i \Theta^{(q)})`
            `\odot` denotes the element-wise multiplication.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=1, label_dim:int=2, \
        activation:bool=False, trainable_q:bool=False, layer:int=2, dropout:float=False, normalization:str='sym', cached: bool=False, absolute_degree: bool=True):
        super(MSGNN_node_classification, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached, absolute_degree=absolute_degree))

        self.Chebs = chebs

        self.Conv = nn.Conv1d(2*hidden, label_dim, kernel_size=1)
        self.dropout = dropout
        self.num_features = num_features
        #self.pi_proj = None  # Lazy initialization for topological feature projection
        self.feature_smoother = torch.nn.Linear(num_features, num_features)

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.Conv.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: Optional[torch.LongTensor]=None, prompt=None, prompt_type=None, PI=None, pi=None, Process_PI = False) -> torch.FloatTensor:
        """
        Making a forward pass of the MSGNN node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features (real and imaginary parts).
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * prompt (optional) - Prompt for GPF or Gprompt.
            * prompt_type (str, optional) - Type of prompt ('GPF' or 'Gprompt').
            * PI (PyTorch Float Tensor, optional) - Topological persistent images, shape (num_nodes, H, W) or (num_nodes, feature_dim).

        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden).
            * **output** (PyTorch FloatTensor) - Log of prob, with shape (num_nodes, num_clusters).
            * **predictions_cluster** (PyTorch LongTensor) - Predicted labels.
            * **prob** (PyTorch FloatTensor) - Probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        # Fuse topological features if provided
        """if PI is not None:
            # Flatten PI if it has spatial dimensions (e.g., from persistent images)
            if PI.dim() > 2:
                PI_flat = PI.view(PI.size(0), -1)
            else:
                PI_flat = PI

            # Lazy initialization of projection layer
            if self.pi_proj is None:
                in_channels = PI_flat.size(1) + real.size(1)
                self.pi_proj = nn.Sequential(
                    nn.Linear(in_channels, self.num_features),
                    nn.BatchNorm1d(self.num_features),
                    nn.ReLU()
                ).to(real.device)

            # Concatenate real features with flattened PI
            real_with_topo = torch.cat([real, PI_flat], dim=1)
            # Project back to original feature dimension
            real = self.pi_proj(real_with_topo)
            # real = F.relu(self.pi_proj(real_with_topo))
            # real = real + F.relu(self.pi_proj(real_with_topo))
            imag = real.clone()

        if Process_PI:
            real = self.feature_smoother(real)
            imag = real.clone()"""

        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real, imag), dim = -1)

        if prompt_type == 'Gprompt':
            x = prompt(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        z = torch.transpose(x[0], 0, 1).clone()
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)

        output = torch.transpose(x[0], 0, 1) # log_prob
        predictions_cluster = torch.argmax(output, dim=1)

        prob = F.softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob


class MSGNNEncoder(nn.Module):
    r"""MSGNN-style encoder that returns node embeddings for signed graphs."""
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=3, label_dim:int=2, \
        activation:bool=False, trainable_q:bool=False, layer:int=2, dropout:float=False, normalization:str='sym', cached: bool=False, absolute_degree: bool=True):
        super(MSGNNEncoder, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached, absolute_degree=absolute_degree))

        self.Chebs = chebs
        self.classifier = torch.nn.LazyLinear(label_dim)  # Lazy initialization for classifier

        self.Conv = nn.Conv1d(2*hidden, label_dim, kernel_size=1)
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.Conv.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: Optional[torch.LongTensor]=None, pi=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.

        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden) for undirected graphs and (num_nodes, 4*hidden) for directed graphs.
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x_msgnn = torch.cat((real, imag), dim = -1)


        # If pi is provided, concatenate it to the node embeddings before the final linear layer
        if pi is not None:
            x_combined = torch.cat((x_msgnn, pi), dim=-1)
        else:
            x_combined = x_msgnn

        z = x_combined.clone()

        # 线性层进行分类
        logits = self.classifier(x_combined) # shape: [N, num_classes]

        # 5. 计算概率和预测
        output = F.log_softmax(logits, dim=1)
        predictions_cluster = torch.argmax(output, dim=1)
        prob = F.softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob
