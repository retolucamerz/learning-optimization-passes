import torch
from torch import nn
from torch.nn import Module, Embedding
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.conv import GINEConv
import numpy as np

from graph_embedding import Inst2vecEmbedding

class NodeEmbedding(Module):
    def __init__(self, emb_dim=200):
        super().__init__()
        self.node_kind_embedding = Embedding(3, emb_dim)
        self.type_embedding = Embedding(1, 200)
        torch.nn.init.xavier_uniform_(self.node_kind_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.type_embedding.weight.data)

        self.encoder = None
        if emb_dim != 200:
            self.encoder = nn.Sequential(
                nn.Linear(200, 50),
                nn.ReLU(inplace=True),
                nn.Linear(50, emb_dim),
            )

    def forward(self, x): # x.shape == (N, 2)
        node_kind_embedded = self.node_kind_embedding(x[:,0])

        node_content_embedded = torch.zeros((*x.shape[:-1], 200)).to(x.device)
        stmt_node_filter = x[:,0]==0
        node_content_embedded[stmt_node_filter] = torch.tensor(np.array([Inst2vecEmbedding.embed_idx(idx) for idx in x[:,1][stmt_node_filter]])).to(x.device)
        node_content_embedded[~stmt_node_filter] = self.type_embedding(x[:,1][~stmt_node_filter])
        if self.encoder is not None:
            node_content_embedded = self.encoder(node_content_embedded)

        return node_kind_embedded + node_content_embedded

class EdgeEmbedding(Module):
    MAX_EDGE_POSITION = 20

    def __init__(self, emb_dim=200):
        super().__init__()

        self.edge_type_embedding = Embedding(3, emb_dim)
        self.position_embedding = Embedding(EdgeEmbedding.MAX_EDGE_POSITION, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_type_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.position_embedding.weight.data)

    def forward(self, edge_attr): # edge_attr.shape == (M, 2)
        return self.edge_type_embedding(edge_attr[...,0]) + self.position_embedding(torch.clamp(edge_attr[...,1], max=EdgeEmbedding.MAX_EDGE_POSITION-1))


class Model(nn.Module):
    def __init__(
        self,
        num_outputs,
        autophase_emb_dim=0,
        hidden_units=20,
        dropout=0.2,
        num_layers=5,
        pool_type="mean",  # global pooling
        conv=GINEConv,
        use_softmax=False,
        use_batchnorm=False,
    ):
        super(Model, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.use_softmax = use_softmax
        self.use_batchnorm = use_batchnorm
        self.pool_type = pool_type
        assert pool_type in ["add", "mean"]

        if autophase_emb_dim>0:
            assert autophase_emb_dim < hidden_units
            self.autophase_net = nn.Sequential(
                nn.Linear(56, 56),
                nn.BatchNorm1d(56),
                nn.ReLU(inplace=True),
                nn.Linear(56, 30),
                nn.ReLU(inplace=True),
                nn.Linear(30, autophase_emb_dim),
            )
        else:
            self.autophase_net = None

        self.node_embedding = NodeEmbedding(emb_dim=hidden_units - autophase_emb_dim)
        self.edge_embedding = EdgeEmbedding(emb_dim=hidden_units)

        self.conv = conv
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs = nn.ModuleList()

        if use_batchnorm:
            self.convs.append(
                conv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.BatchNorm1d(hidden_units),
                        nn.ReLU(),
                        nn.Linear(hidden_units, hidden_units),
                    ),
                )
            )
        else:
            self.convs.append(
                conv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.ReLU(),
                        nn.Linear(hidden_units, hidden_units),
                    ),
                )
            )
        self.bns.append(nn.BatchNorm1d(hidden_units))
        self.fcs.append(nn.Linear(hidden_units, num_outputs))
        self.fcs.append(nn.Linear(hidden_units, num_outputs))

        for i in range(self.num_layers - 1):
            self.convs.append(
                conv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.BatchNorm1d(hidden_units),
                        nn.ReLU(),
                        nn.Linear(hidden_units, hidden_units),
                    ),
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_units))
            self.fcs.append(nn.Linear(hidden_units, num_outputs))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, self.conv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data, **kwargs):
        x = self.node_embedding(data.x)
        if self.autophase_net is not None:
            autophase = data.autophase
            if len(data.autophase.shape)<2:
                autophase = autophase[None]
            
            autophase = self.autophase_net(autophase)
            if data.batch is None:
                autophase = autophase.expand(x.shape[0], autophase.shape[-1])
            else:
                autophase = autophase[data.batch]
            x = torch.cat((x, autophase), dim=-1)

        edge_attr = self.edge_embedding(data.edge_attr)
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, data.edge_index, edge_attr)
            # x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)

        out: Tensor = None # type: ignore
        for i, x in enumerate(outs):
            if self.pool_type == "add":
                x = global_add_pool(x, data.batch)
            elif self.pool_type == "mean":
                x = global_mean_pool(x, data.batch)

            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        if self.use_softmax:
            return F.softmax(out, dim=-1)
        else:
            return out
