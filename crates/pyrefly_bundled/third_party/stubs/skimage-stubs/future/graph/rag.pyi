import math
from collections.abc import Mapping, Sequence
from typing import Callable, Literal

import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage as ndi, sparse

from ..._shared.version_requirements import require

def _edge_generator_from_csr(csr_matrix): ...
def min_weight(graph: RAG, src: int, dst: int, n: int) -> Mapping: ...
def _add_edge_filter(values, graph): ...

class RAG(nx.Graph):
    def __init__(self, label_image=None, connectivity=1, data=None, **attr): ...
    def merge_nodes(
        self,
        src: int,
        dst: int,
        weight_func: Callable = ...,
        in_place: bool = True,
        extra_arguments: Sequence = [],
        extra_keywords: Mapping = {},
    ) -> int: ...
    def add_node(self, n, attr_dict=None, **attr): ...
    def add_edge(self, u, v, attr_dict=None, **attr): ...
    def copy(self): ...
    def fresh_copy(self): ...
    def next_id(self) -> int: ...
    def _add_node_silent(self, n): ...

def rag_mean_color(
    image,
    labels,
    connectivity: int = 2,
    mode: Literal["distance", "similarity"] = "distance",
    sigma: float = 255.0,
) -> RAG: ...
def rag_boundary(labels, edge_map, connectivity=2): ...
@require("matplotlib", ">=3.0.3")
def show_rag(
    labels,
    rag: RAG,
    image,
    border_color="black",
    edge_width: float = 1.5,
    edge_cmap="magma",
    img_cmap="bone",
    in_place: bool = True,
    ax=None,
): ...
