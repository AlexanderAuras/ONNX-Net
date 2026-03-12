"""Module defining a Directed Acyclic Graph (DAG) data structure."""  # noqa: EXE002

from __future__ import annotations

import io
import itertools
from typing import TYPE_CHECKING, Any, Literal, cast, override

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pydot
import scipy as sp


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
if TYPE_CHECKING:
    import numpy.typing as npt


class ForeignNodeError(Exception):
    """Exception raised when a node does not belong to the graph it is being used with."""


class ForeignEdgeError(Exception):
    """Exception raised when an edge does not belong to the graph it is being used with."""


class DAG[N, E]:  # noqa: PLR0904
    """Directed Acyclic Graph (DAG) data structure."""

    def __init__(self) -> None:
        """Initialize an empty DAG."""
        super().__init__()
        self.__nodes: list[Node[N, E]] = []

    @property
    def nodes(self) -> list[Node[N, E]]:
        """Get a list of all nodes in the DAG.

        Returns:
            A copy of the list of nodes in the DAG.

        """
        return self.__nodes.copy()

    @property
    def edges(self) -> list[Edge[N, E]]:
        """Get a list of all edges in the DAG.

        Returns:
            A copy of the list of edges in the DAG.

        """
        return [edge for node in self.__nodes for edge in node.outgoing_edges]

    def add_node(self, value: N) -> Node[N, E]:
        """Add a node with the given value to the DAG.

        Args:
            value: The value to assign to the new node.

        Returns:
            The newly created node.

        """
        node = Node(self, value)
        self.__nodes.append(node)
        return node

    def remove_nodes(self, *args: Node[N, E]) -> None:
        """Remove nodes from the DAG.

        Raises:
            ForeignNodeError: If a node does not belong to this graph.

        """
        for node in args:
            if node not in self.__nodes or node.graph != self:
                raise ForeignNodeError
            for edge in list(node.incoming_edges):
                getattr(edge.source, "_Node__outgoing_edges").remove(edge)  # noqa: B009
            for edge in list(node.outgoing_edges):
                getattr(edge.target, "_Node__incoming_edges").remove(edge)  # noqa: B009
            self.__nodes.remove(node)

    def copy(self) -> DAG[N, E]:
        """Create a copy of the DAG.

        Returns:
            A new DAG that is a copy of this DAG.
        """
        new_dag = DAG[N, E]()
        node_map: dict[Node[N, E], Node[N, E]] = {}
        for node in self.__nodes:
            node_map[node] = new_dag.add_node(node.value)
        for edge in self.edges:
            _ = new_dag.add_edge(
                node_map[edge.source],
                node_map[edge.target],
                edge.value,
                check=False,
            )
        return new_dag

    def dfs(
        self,
        start: Node[N, E],
        *,
        reverse: bool = False,
    ) -> Generator[Node[N, E]]:
        """Perform a depth-first search (DFS) starting from the given node.

        Args:
            start: The node to start the DFS from.
            reverse: If True, perform the DFS in reverse order.

        Raises:
            ForeignNodeError: If the start node does not belong to this graph.

        Yields:
            The nodes visited during the DFS.

        """
        if start not in self.__nodes or start.graph != self:
            raise ForeignNodeError
        visited = set()

        def _dfs(node: Node[N, E]) -> Generator[Node[N, E]]:
            visited.add(node)
            yield node
            for edge in node.incoming_edges if reverse else node.outgoing_edges:
                if (edge.source if reverse else edge.target) not in visited:
                    yield from _dfs(edge.source if reverse else edge.target)

        yield from _dfs(start)

    def bfs(
        self,
        start: Node[N, E],
        *,
        reverse: bool = False,
    ) -> Generator[Node[N, E]]:
        """Perform a breadth-first search (BFS) starting from the given node.

        Args:
            start: The node to start the BFS from.
            reverse: If True, perform the BFS in reverse order.

        Raises:
            ForeignNodeError: If the start node does not belong to this graph.

        Yields:
            The nodes visited during the BFS.

        """
        if start not in self.__nodes or start.graph != self:
            raise ForeignNodeError
        visited = set()
        queue = [start]
        while len(queue) > 0:
            node = queue.pop(0)
            visited.add(node)
            for edge in node.incoming_edges if reverse else node.outgoing_edges:
                if (edge.source if reverse else edge.target) not in visited:
                    queue.append(edge.source if reverse else edge.target)  # noqa: PERF401
            yield node

    def grouped_topological_sorting(
        self,
        key: Callable[[Node[N, E]], Any] | None = None,
    ) -> Generator[list[Node[N, E]]]:
        """Returns groups of nodes in topological order.

        Args:
            key: An optional function to sort nodes within the same group, for a unique ordering.

        Yields:
            The group of nodes coming in the same place for a topological sorting.
        """
        in_degree_map: dict[Node[N, E], int] = {node: len(node.incoming_edges) for node in self.__nodes}
        candidates = {node for node in self.__nodes if in_degree_map[node] == 0}
        while len(candidates) > 0:
            group: list[Node[N, E]] = []
            for node in candidates.copy():
                if in_degree_map[node] == 0:
                    group.append(node)
                    candidates.remove(node)
                    for edge in node.outgoing_edges:
                        in_degree_map[edge.target] -= 1
                        if in_degree_map[edge.target] == 0:
                            candidates.add(edge.target)
            yield sorted(group, key=key) if key is not None else group

    def topological_sorting(self, key: Callable[[Node[N, E]], Any] | None = None) -> Generator[Node[N, E]]:
        """Returns nodes in topological order.

        Args:
            key: An optional function to sort nodes within the same group, for a unique ordering.

        Yields:
            The nodes in topological order.
        """
        yield from itertools.chain.from_iterable(self.grouped_topological_sorting(key=key))

    def add_edge(
        self,
        start: Node[N, E],
        end: Node[N, E],
        value: E,
        *,
        check: bool = True,
    ) -> Edge[N, E]:
        """Add a directed edge from `start` to `end` with the given value.

        Args:
            start: The starting node of the edge.
            end: The ending node of the edge.
            value: The value to assign to the edge.
            check: If True, check for cycles before adding the edge.

        Raises:
            ForeignNodeError: If the start or end node does not belong to this graph.

        Returns:
            The created edge.

        """
        if start not in self.__nodes or start.graph != self:
            raise ForeignNodeError
        if end not in self.__nodes or end.graph != self:
            raise ForeignNodeError
        if check:
            for node in self.bfs(end):
                if node == start:
                    pass  # raise ValueError("Adding this edge would create a cycle")
        edge = Edge[N, E](self, start, end, value)
        getattr(start, "_Node__outgoing_edges").append(edge)  # noqa: B009
        getattr(end, "_Node__incoming_edges").append(edge)  # noqa: B009
        return edge

    def remove_edges(self, *args: Edge[N, E]) -> None:
        """Remove edges from the DAG.

        Raises:
            ForeignEdgeError: If the edge does not belong to this graph.

        """
        for edge in args:
            if edge.source not in self.__nodes or edge.source.graph != self:
                raise ForeignEdgeError
            if edge.target not in self.__nodes or edge.target.graph != self:
                raise ForeignEdgeError
            getattr(edge.source, "_Node__outgoing_edges").remove(edge)  # noqa: B009
            getattr(edge.target, "_Node__incoming_edges").remove(edge)  # noqa: B009

    @override
    def __repr__(self) -> str:
        string = "digraph {\n"
        node_id_map: dict[Node[N, E], int] = {}
        for node in self.__nodes:
            node_id_map[node] = len(node_id_map)
            string += (
                f"    node{node_id_map[node]}"
                + f"{'[label=' + chr(0x22) + str(node.value) + chr(0x22) + ']' if node.value is not None else ''}\n"
            )
        for edge in self.edges:
            string += (
                f"    node{node_id_map[edge.source]}->node{node_id_map[edge.target]}"
                + f"{'[label=' + chr(0x22) + str(edge.value) + chr(0x22) + ']' if edge.value is not None else ''}\n"
            )
        string += "}"
        return string

    @override
    def __str__(self) -> str:
        return f"Digraph({len(self.__nodes)} nodes, {len(self.edges)} edges)"

    @classmethod
    def from_networkx(
        cls,
        graph: nx.DiGraph[Any],
        node_attr_name: str | None = "value",
        edge_attr_name: str | None = "value",
        *,
        check: bool = True,
    ) -> DAG[Any, Any]:
        """Create a DAG from a NetworkX DiGraph.

        Args:
            graph: The NetworkX DiGraph to convert.
            node_attr_name: The attribute name to use for node values.
            edge_attr_name: The attribute name to use for edge values.
            check: If True, check for cycles before adding edges.

        Returns:
            The created DAG.

        """
        digraph = cast("DAG[Any, Any]", cls())
        id_node_map: dict[int, Node[Any, Any]] = {}
        for node_id, data in graph.nodes(data=True):
            id_node_map[node_id] = digraph.add_node(
                data[node_attr_name] if node_attr_name else data,
            )
        for start_id, end_id, data in graph.edges(data=True):
            _ = digraph.add_edge(
                id_node_map[start_id],
                id_node_map[end_id],
                data[edge_attr_name] if edge_attr_name else data,
                check=check,
            )
        return digraph

    def to_networkx(self, *, use_ids: bool = True) -> nx.DiGraph[Any]:
        """Convert the DAG to a NetworkX DiGraph.

        Args:
            use_ids: If True, use newly generated node IDs as the NetworkX
                     node identifiers instead of the nodes `value`-attribute.

        Returns:
            The converted NetworkX DiGraph.

        """
        graph = nx.DiGraph()
        node2id_map: dict[Node[N, E], int] = {}
        for node in self.__nodes:
            if use_ids:
                if node not in node2id_map:
                    node2id_map[node] = len(node2id_map)
                graph.add_node(node2id_map[node], value=node.value)
            else:
                graph.add_node(node.value)
        for edge in self.edges:
            if use_ids:
                graph.add_edge(
                    node2id_map[edge.source],
                    node2id_map[edge.target],
                    value=edge.value,
                )
            else:
                graph.add_edge(edge.source.value, edge.target.value, value=edge.value)
        return graph

    def render(
        self,
        node_labeler: Callable[[N], str] | None = None,
        edge_labeler: Callable[[E], str] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Render the DAG as an image.

        Args:
            node_labeler: A function to label the nodes.
            edge_labeler: A function to label the edges.

        Returns:
            A NumPy array representing the rendered image of the DAG.

        """
        graph = pydot.Dot(graph_type="digraph")
        node2pydot_map: dict[Node[N, E], pydot.Node] = {}
        for node in self.__nodes:
            node2pydot_map[node] = pydot.Node(
                str(id(node)),
                label=node_labeler(node.value) if node_labeler is not None else None,
            )
            graph.add_node(node2pydot_map[node])
        for edge in self.edges:
            graph.add_edge(
                pydot.Edge(
                    node2pydot_map[edge.source],
                    node2pydot_map[edge.target],
                    label=edge_labeler(edge.value) if edge_labeler is not None else None,
                ),
            )
        buffer = io.BytesIO(graph.create_png())  # pyright: ignore [reportAttributeAccessIssue]
        return plt.imread(buffer)

    def collapse_edge(
        self,
        edge: Edge[N, E],
        node_value_merger: Callable[[N, N], N],
        in_edge_modifier: Callable[[E], E] | None = None,
        out_edge_modifier: Callable[[E], E] | None = None,
        *,
        check: bool = True,
    ) -> None:
        """Collapse an edge in the DAG by merging the source and target nodes.

        Args:
            edge: The edge to collapse.
            node_value_merger: A function to merge the values of the source and target nodes.
            in_edge_modifier: A function to modify the values of incoming edges to the target node.
            out_edge_modifier: A function to modify the values of outgoing edges from the source node.
            check: Whether to check for the existence of the nodes in the graph.

        Raises:
            ForeignEdgeError: If the edge's source or target node is not in the graph.

        """
        if edge.source not in self.__nodes or edge.source.graph != self:
            raise ForeignEdgeError
        if edge.target not in self.__nodes or edge.target.graph != self:
            raise ForeignEdgeError
        for in_edge in edge.target.incoming_edges:
            if in_edge.source == edge.source:
                continue
            new_value = in_edge_modifier(in_edge.value) if in_edge_modifier is not None else in_edge.value
            self.add_edge(in_edge.source, edge.source, new_value, check=check)
        for out_edge in edge.target.outgoing_edges:
            new_value = out_edge_modifier(out_edge.value) if out_edge_modifier is not None else out_edge.value
            self.add_edge(edge.source, out_edge.target, new_value, check=check)
        self.remove_nodes(edge.target)
        new_value = node_value_merger(edge.source.value, edge.target.value)
        edge.source.value = new_value

    def adjacency_matrix(self) -> sp.sparse.coo_array:
        """
        Generate the adjacency matrix of the DAG.

        Returns:
            A sparse COO array representing the adjacency matrix of the DAG.
        """
        data, row, col = [], [], []
        sorted_nodes = list(self.topological_sorting(key=str))
        for i, node in enumerate(sorted_nodes):
            for edge in node.outgoing_edges:
                j = sorted_nodes.index(edge.target)
                # if not allow_parallel_edges and len(data) > 0 and row[-1] == i and col[-1] == j:
                #    continue
                data.append(1)
                row.append(i)
                col.append(j)
        return sp.sparse.coo_array((data, (row, col)), shape=(len(self.__nodes), len(self.__nodes)))

    def degree_matrix(
        self,
        *,
        direction: Literal["in", "out"] = "out",
        insert_self_loops: bool = False,
    ) -> sp.sparse.coo_array:
        """
        Generates a diagonal matrix with the in- or out-degrees of a all nodes.

        Args:
            direction: Specifies whether to compute the in-degrees or out-degrees of the nodes.

        Returns:
            A sparse DIA array representing the degree matrix of the DAG.
        """
        degrees = []
        for node in self.topological_sorting(key=str):
            if insert_self_loops:
                degrees.append(
                    max(len(node.outgoing_edges) if direction == "out" else len(node.incoming_edges), 1),
                )
            else:
                degrees.append(len(node.outgoing_edges) if direction == "out" else len(node.incoming_edges))
        return sp.sparse.dia_array((degrees, [0]), shape=(len(self.__nodes), len(self.__nodes)))

    def generate_position_encodings(self, k_frac: float = 0.2) -> npt.NDArray[np.float64]:
        """
        Generate position encodings for the nodes in the DAG.

        Args:
            k_frac: Fraction of singular values/vectors to retain for the encoding.

        Returns:
            A numpy array with shape (num_nodes, encoding_dim) representing the encodings of the nodes in the DAG.
        """
        adjacency = self.adjacency_matrix()
        out_degrees = self.degree_matrix(direction="out", insert_self_loops=True)
        in_degrees = self.degree_matrix(direction="in", insert_self_loops=True)
        # Use "SVDFormer" structure matrix with "Edge-augmented Graph Transformer"'s SVD approach
        matrix = out_degrees.power(-0.5) @ (adjacency + sp.sparse.identity(len(self.__nodes))) @ in_degrees.power(-0.5)
        U, S, Vt = sp.sparse.linalg.svds(matrix, k=int(k_frac * (len(self.__nodes) - 1)), which="LM")  # noqa: N806
        idx = np.argsort(S)[::-1]
        U, S, Vt = U[:, idx], S[idx], Vt[idx]  # noqa: N806 # pyright: ignore [reportConstantRedefinition]
        V = Vt.T  # noqa: N806
        signs = np.sign(U.sum(axis=0))
        signs[signs == 0] = 1
        U *= signs  # noqa: N806 # pyright: ignore [reportConstantRedefinition]
        V *= signs  # noqa: N806 # pyright: ignore [reportConstantRedefinition]
        U /= np.linalg.norm(U, axis=0, keepdims=True)  # noqa: N806 # pyright: ignore [reportConstantRedefinition]
        V /= np.linalg.norm(V, axis=0, keepdims=True)  # noqa: N806 # pyright: ignore [reportConstantRedefinition]
        enc = np.concatenate([U, V], axis=1)
        return enc


class Node[N, E]:
    """A node in a Directed Acyclic Graph (DAG)."""

    def __init__(self, graph: DAG[N, E], value: N) -> None:
        """Initialize a node with the given value in the specified graph."""
        super().__init__()
        self.__graph = graph
        self.__value = value
        self.__incoming_edges = []
        self.__outgoing_edges = []

    @property
    def graph(self) -> DAG[N, E]:
        """Get the graph to which this node belongs.

        Returns:
            The graph containing this node.

        """
        return self.__graph

    @property
    def value(self) -> N:
        """Get the value of the node.

        Returns:
            The value of the node.

        """
        return self.__value

    @value.setter
    def value(self, new_value: N) -> None:
        self.__value = new_value

    @property
    def incoming_edges(self) -> list[Edge[N, E]]:
        """Get the incoming edges of the node.

        Returns:
            A copy of the list of incoming edges.

        """
        return self.__incoming_edges.copy()

    @property
    def outgoing_edges(self) -> list[Edge[N, E]]:
        """Get the outgoing edges of the node.

        Returns:
            A copy of the list of outgoing edges.

        """
        return self.__outgoing_edges.copy()

    @override
    def __str__(self) -> str:
        return f"({self.value})"


class Edge[N, E]:
    """An edge in a Directed Acyclic Graph (DAG)."""

    def __init__(
        self,
        graph: DAG[N, E],
        start: Node[N, E],
        end: Node[N, E],
        value: E,
    ) -> None:
        """Initialize an edge with the given start and end nodes and value.

        Args:
            graph: The graph containing this edge.
            start: The source node of the edge.
            end: The target node of the edge.
            value: The value of the edge.

        """
        super().__init__()
        self.__graph = graph
        self.__start = start
        self.__end = end
        self.__value = value

    @property
    def graph(self) -> DAG[N, E]:
        """Get the graph to which this edge belongs.

        Returns:
            The graph containing this edge.

        """
        return self.__graph

    @property
    def value(self) -> E:
        """Get the value of the edge.

        Returns:
            The value of the edge.

        """
        return self.__value

    @value.setter
    def value(self, new_value: E) -> None:
        self.__value = new_value

    @property
    def source(self) -> Node[N, E]:
        """Get the source node of the edge.

        Returns:
            The source node of the edge.

        """
        return self.__start

    @property
    def target(self) -> Node[N, E]:
        """Get the target node of the edge.

        Returns:
            The target node of the edge.

        """
        return self.__end

    @override
    def __str__(self) -> str:
        return f"({self.source.value}) --[{self.value}]--> ({self.target.value})"
