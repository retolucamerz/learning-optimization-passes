import torch
import numpy as np
import pickle
from inst2vec import inst2vec_preprocess
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from consts import AUTOPHASE_MEAN, AUTOPHASE_STD

_PICKLED_VOCABULARY = "inst2vec/dictionary.pickle"
_PICKLED_EMBEDDINGS = "inst2vec/embeddings.pickle"

with open(str(_PICKLED_VOCABULARY), "rb") as f:
    vocab = pickle.load(f)

with open(str(_PICKLED_EMBEDDINGS), "rb") as f:
    embeddings = pickle.load(f)


class Inst2vecEmbedding:
    def __init__(self, ir: str):
        self.structs = inst2vec_preprocess.GetStructTypes(ir)

    def preprocess(self, stmt: str):
        for struct, definition in self.structs.items():
            stmt = stmt.replace(struct, definition)
        stmt = inst2vec_preprocess.preprocess([[stmt]])[0][0]
        if not stmt:
            return None
        return inst2vec_preprocess.PreprocessStatement(stmt[0])

    def discretize(self, stmt: str):
        token = self.preprocess(stmt)
        return vocab.get(token, None)

    @staticmethod
    def embed_idx(idx: int):
        if idx is None or idx >= Inst2vecEmbedding.num_values():
            idx = vocab["!UNK"]
        return embeddings[idx]

    def embed(self, stmt: str):
        idx = self.discretize(stmt)
        return self.embed_idx(idx)

    @staticmethod
    def num_values():
        return len(vocab)


# type==0 (stmt) -> Inst2vec
# type==1 (type) -> attrs["text"]
# type==2 (const) -> ???

# {attrs["text"] for id, attrs in G.nodes(data=True) if attrs["type"]==0}
stmt_types = {
    'unreachable': 0,
    '; undefined function': 1,
    'zext': 2,
    'sub': 3,
    'phi': 4,
    'store': 5,
    'udiv': 6,
    'getelementptr': 7,
    'load': 8,
    'call': 9,
    'mul': 10,
    'add': 11,
    'trunc': 12,
    'br': 13,
    'ptrtoint': 14,
    'fadd': 15,
    'icmp': 16,
    '[external]': 17,
    'bitcast': 18,
    'ret': 19,
    'sdiv': 20,
    'sext': 21,
    'sitofp': 22,
    'alloca': 23,
    'fcmp': 24,
    'select': 25,
    'and': 26,
    'or': 31,
    'shl': 28,
    'lshr': 29,
    'ashr': 30,
    'xor': 32,
    'switch': 33,
    'srem': 34,
    'fdiv': 35,
    'urem': 36,
    'fmul': 37,
    'fptosi': 38,
    'fptrunc': 39,
    'fsub': 40,
    'uitofp': 41,
    'fpext': 42,
    'fptoui': 43,
    'fneg': 44,
    'inttoptr': 45,
    'insertelement': 46,
    'extractelement': 47,
    'shufflevector': 48,
}

class LlvmGraphRepresentationWithInst2vec:
    def __init__(self, observation):
        self.ir = observation["Ir"]
        self.autophase = observation["Autophase"]
        self.G = observation["Programl"]
        self.instr_embedding = Inst2vecEmbedding(self.ir)

    def encode_stmt_nodes(self, nodeattrs_by_id):
        assert all(len(nodeattrs["features"]["full_text"])==1 for _, nodeattrs in nodeattrs_by_id.items() if "features" in nodeattrs)
        stmt_by_node_id = {
            id: nodeattrs["features"]["full_text"][0] for id, nodeattrs in nodeattrs_by_id.items()
            if "features" in nodeattrs
        }
        disc_feat_by_node_id = {
            id: self.instr_embedding.discretize(stmt) for id, stmt in stmt_by_node_id.items()
        }

        # no fulltext -> "[external]"
        assert all(attrs["text"] == "[external]" for _, attrs in self.G.nodes(data=True) if "features" not in attrs)
        node_ids_without_fulltext = [id for id, node_repr in self.G.nodes(data=True) if "features" not in node_repr]
        disc_feat_by_node_id.update({id: Inst2vecEmbedding.num_values() for id in node_ids_without_fulltext})

        # no node feature embedding
        disc_feat_by_node_id.update({
            id: Inst2vecEmbedding.num_values() + 1 + stmt_types[nodeattrs_by_id[id]["text"]]
            for id, disc_feat in disc_feat_by_node_id.items()
            if disc_feat is None
        })

        return disc_feat_by_node_id

    def encode_type_const_nodes(self, nodeattrs_by_id):
        # TODO somehow encode type
        return {id: 0 for id, _ in nodeattrs_by_id.items()}

    def as_discrete_graph(self):
        disc_feat_by_node_id = dict()
        disc_feat_by_node_id.update(self.encode_stmt_nodes({
            id: node_attrs for id, node_attrs in self.G.nodes(data=True) if node_attrs["type"]==0
        }))
        disc_feat_by_node_id.update(self.encode_type_const_nodes({
            id: node_attrs for id, node_attrs in self.G.nodes(data=True) if node_attrs["type"]!=0
        }))

        graph = nx.MultiDiGraph()
        for id, disc_feat in disc_feat_by_node_id.items():
            attrs = self.G.nodes(data=True)[id]
            if "features" in attrs:
                full_text = attrs["features"]["full_text"][0]
            else:
                full_text = attrs["text"]
            graph.add_node(id, x=(attrs["type"], disc_feat), full_text=full_text, text=attrs["text"])
        for e1, e2, edge_attrs in self.G.edges(data=True):
            graph.add_edge(e1, e2, edge_attr=(edge_attrs["flow"], edge_attrs["position"]))
        pyg_graph = from_networkx(graph)
        pyg_graph.ir = self.ir

        autophase = torch.as_tensor(np.array(self.autophase))
        autophase = (autophase - AUTOPHASE_MEAN) / AUTOPHASE_STD
        pyg_graph.autophase = autophase[None]

        return pyg_graph
