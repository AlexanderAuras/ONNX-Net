# noqa: EXE002
"""Utility functions for encoding and decoding ONNX graphs."""

from __future__ import annotations

import enum
from itertools import chain
import operator
from typing import Any, override

import onnx

from onnxnet.process.dag import DAG, Node


class Operation(enum.StrEnum):
    """Enum for different types of operations in an ONNX graph."""

    INPUT = "Input"
    PARAM = "Parameter"
    CONST = "Constant"
    OUTPUT = "Output"

    IDENTITY = "Identity"
    ADD = "Add"
    MUL = "Mul"
    SLICE = "Slice"
    CONCAT = "Concat"
    RESHAPE = "Reshape"
    TRANSPOSE = "Transpose"
    SPLIT = "Split"
    UNSQUEEZE = "Unsqueeze"
    RELU = "Relu"
    HARD_SWISH = "HardSwish"
    SOFTMAX = "Softmax"
    PAD = "Pad"
    CONV = "Conv"
    MAXPOOL = "MaxPool"
    AVGPOOL = "AveragePool"
    GLOBAL_AVGPOOL = "GlobalAveragePool"
    GEMM = "Gemm"
    MATMUL = "MatMul"
    MEAN = "ReduceMean"
    SUM = "ReduceSum"
    GATHER = "Gather"
    BATCHNORM = "BatchNormalization"
    LAYERNORM = "LayerNormalization"
    INSTANCENORM = "InstanceNormalization"

    ADD_CONST = "AddConst"
    LINEAR = "Linear"


OP_ABBR_MAP = {
    Operation.INPUT: "Input",
    Operation.PARAM: "Param",
    Operation.CONST: "Const",
    Operation.OUTPUT: "Output",
    Operation.IDENTITY: "Identity",
    Operation.CONV: "Conv",
    Operation.RELU: "Relu",
    Operation.HARD_SWISH: "HardSwish",
    Operation.ADD: "Add",
    Operation.MUL: "Mul",
    Operation.SLICE: "Slice",
    Operation.MAXPOOL: "MaxPool",
    Operation.AVGPOOL: "AvgPool",
    Operation.GLOBAL_AVGPOOL: "GlobalAvgPool",
    Operation.CONCAT: "Concat",
    Operation.MEAN: "Mean",
    Operation.GEMM: "GEMM",
    Operation.BATCHNORM: "BatchNorm",
    Operation.LAYERNORM: "LayerNorm",
    Operation.INSTANCENORM: "InstanceNorm",
    Operation.RESHAPE: "Reshape",
    Operation.MATMUL: "MatMul",
    Operation.GATHER: "Gather",
    Operation.TRANSPOSE: "Transpose",
    Operation.SPLIT: "Split",
    Operation.UNSQUEEZE: "Unsqueeze",
    Operation.SUM: "Sum",
    Operation.PAD: "Pad",
    Operation.SOFTMAX: "Softmax",
    Operation.ADD_CONST: "AddConst",
    Operation.LINEAR: "Linear",
}


class ONNXEdge:
    """Class representing an edge in an ONNX graph."""

    def __init__(self, name: str, in_index: int, shape: tuple[int, ...] | None) -> None:
        """Initialize an ONNX edge.

        Args:
            name: The name of the tensor this edge represents.
            in_index: The input index of the edge at the target node.
            shape: The shape of the tensor this edge represents.

        """
        super().__init__()
        self.name = name
        self.index = in_index
        self.shape = shape

    @override
    def __str__(self) -> str:
        return str(self.shape)


def merge_repeated_tuple(x: Any) -> Any:  # noqa: ANN401
    """Combines tuples with identical arguments into a single value.

    Args:
        x: The input tuple.

    Returns:
        The only unique entry in the tuple, or the original tuple if there are multiple unique entries.
    """
    if not isinstance(x, tuple) or len(x) == 0:
        return x
    for entry in x:
        if entry != x[0]:
            return x
    return x[0]


class ONNXNode:
    """Class representing a node in an ONNX graph."""

    def __init__(self, name: str, index: int, operation: Operation, attributes: dict[str, Any] | None = None) -> None:
        """Initialize an ONNX node.

        Args:
            name: The name of the node.
            index: The index in the onnx file of this node.
            operation: The operation this node represents.
            attributes: The attributes of the node.

        """
        super().__init__()
        self.name = name
        self.index = index
        self.operation = operation
        self.attributes = attributes if attributes is not None else {}

    @override
    def __str__(self) -> str:  # noqa: C901, PLR0911, PLR0912
        match self.operation:
            case Operation.INPUT:
                return f"Input()  # ({','.join(map(str, self.attributes['shape']))})"
            case Operation.CONST:
                return f"Const(({','.join(map(str, self.attributes['shape']))}))"
            case Operation.PARAM:
                return f"Parameter(({','.join(map(str, self.attributes['shape']))}))"
            case Operation.OUTPUT:
                return f"Output()  # ({','.join(map(str, self.attributes['shape']))})"
            case Operation.AVGPOOL:
                return (
                    f"AvgPool{len(self.attributes['kernel_shape'])}D("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"size={merge_repeated_tuple(self.attributes['kernel_shape'])}",
                                f"stride={merge_repeated_tuple(self.attributes.get('strides', 1))}"
                                if merge_repeated_tuple(self.attributes.get("strides", 1)) != 1
                                else "",
                                f"dilation={merge_repeated_tuple(self.attributes.get('dilations', 1))}"
                                if merge_repeated_tuple(self.attributes.get("dilations", 1)) != 1
                                else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.CONCAT:
                return (
                    "Concat("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes['axis']}" if self.attributes.get("axis", 0) != 0 else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.CONV:
                return (
                    # BUG: Convolution dimension detection does not always work for some reason
                    # f"Conv{len(self.attributes['kernel_shape'])}D("
                    (
                        f"Conv{len(self.attributes['kernel_shape'])}D("
                        if "kernel_shape" in self.attributes
                        else "ConvXD("
                    )
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"inc={merge_repeated_tuple(self.attributes['in_channels'])}",
                                f"outc={merge_repeated_tuple(self.attributes['out_channels'])}",
                                # BUG: Convolution size detection does not always work for some reason
                                # f"size={merge_repeated_tuple(self.attributes['kernel_shape'])}",
                                f"size={merge_repeated_tuple(self.attributes['kernel_shape'])}"
                                if "kernel_shape" in self.attributes
                                else "size=???",
                                f"stride={merge_repeated_tuple(self.attributes.get('strides', 1))}"
                                if merge_repeated_tuple(self.attributes.get("strides", 1)) != 1
                                else "",
                                f"dilation={merge_repeated_tuple(self.attributes.get('dilations', 1))}"
                                if merge_repeated_tuple(self.attributes.get("dilations", 1)) != 1
                                else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.GATHER:
                return (
                    "Gather("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes.get('axis', 0)}" if self.attributes.get("axis", 0) != 0 else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.MAXPOOL:
                return (
                    f"MaxPool{len(self.attributes['kernel_shape'])}D("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                # BUG: Convolution size detection does not always work for some reason
                                # f"size={merge_repeated_tuple(self.attributes['kernel_shape'])}",
                                f"size={merge_repeated_tuple(self.attributes['kernel_shape'])}"
                                if "kernel_shape" in self.attributes
                                else "size=3",
                                f"stride={merge_repeated_tuple(self.attributes.get('strides', 1))}"
                                if merge_repeated_tuple(self.attributes.get("strides", 1)) != 1
                                else "",
                                f"dilation={merge_repeated_tuple(self.attributes.get('dilations', 1))}"
                                if merge_repeated_tuple(self.attributes.get("dilations", 1)) != 1
                                else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.MEAN:
                return (
                    "Mean("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes.get('axes', 'all')}"
                                if self.attributes.get("axes", "all") != "all"
                                else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.SUM:
                return (
                    "Sum("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes.get('axes', 'all')}"
                                if self.attributes.get("axes", "all") != "all"
                                else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.PAD:
                return (
                    "Pad("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes.get('mode', 'constant')}"
                                if self.attributes.get("mode", "constant") != "constant"
                                else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.SOFTMAX:
                return (
                    "Softmax("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes.get('axis', 0)}" if self.attributes.get("axis", 0) != 0 else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.SPLIT:
                return (
                    "Split("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes.get('axis', 0)}" if self.attributes.get("axis", 0) != 0 else "",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.TRANSPOSE:
                return (
                    "Transpose("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes['perm']}",
                            ],
                        ),
                    )
                    + ")"
                )
            case Operation.LINEAR:
                return (
                    "Linear("
                    + ",".join(
                        filter(
                            lambda x: len(x) > 0,
                            [
                                f"{self.attributes['in_features']}",
                                f"{self.attributes['out_features']}",
                            ],
                        ),
                    )
                    + ")"
                )
            case _:
                return OP_ABBR_MAP[self.operation]


def encode_graph(graph: DAG[ONNXNode, ONNXEdge], *, return_node_ids: bool = False) -> str | tuple[str, list[int]]:  # noqa: C901, PLR0912
    """Encode a DAG of ONNX nodes and edges into a string representation.

    Args:
        graph: The DAG to encode.
        return_node_ids: Whether to return the mapping of node to character index in the string representation as well.

    Returns:
        The string representation of the encoded DAG.
    """
    chain_graph = DAG[list[ONNXNode], ONNXEdge]()
    node_map: dict[Node[ONNXNode, ONNXEdge], Node[list[ONNXNode], ONNXEdge]] = {}
    # Recreate graph with lists as nodes
    for node in graph.nodes:
        node_map[node] = chain_graph.add_node([node.value])
    for edge in graph.edges:
        chain_graph.add_edge(start=node_map[edge.source], end=node_map[edge.target], value=edge.value, check=False)
    # Fuse sequential nodes
    while True:
        for edge in chain_graph.edges:
            if len(edge.source.outgoing_edges) == 1 and len(edge.target.incoming_edges) == 1:
                chain_graph.collapse_edge(edge, operator.add, check=False)
                break
        else:
            break
    orig_node_ids = {node.value: i for i, node in enumerate(graph.topological_sorting(key=str))}

    # Build string representation from list graph
    string = ""
    node_id_map: dict[tuple[Node[list[ONNXNode], ONNXEdge], int], int] = {}
    for node in chain_graph.nodes:
        for i in {x.value.index for x in node.outgoing_edges}:
            node_id_map[node, i] = len(node_id_map) + 1
    char_node_ids = []
    for node in chain_graph.topological_sorting(key=str):
        if len(node.outgoing_edges) > 0:
            unique_out_edges = set()
            for edge in node.outgoing_edges:
                if edge.value.index not in {x.value.index for x in unique_out_edges}:
                    unique_out_edges.add(edge)
            # string += ",".join([f"value{node_id_map[node, x.value.index]}" + " = " + (f"[{','.join(map(str, x.value.shape))}]" if x.value.shape is not None else "") for x in unique_out_edges])  # noqa: E501
            substr = ",".join([f"value{node_id_map[node, x.value.index]}" for x in unique_out_edges]) + " = "
            char_node_ids.extend([orig_node_ids[node.value[0]]] * len(substr))
            string += substr
        else:
            string += "output = "
            char_node_ids.extend([orig_node_ids[node.value[0]]] * 9)
        if len(node.value) > 1:
            string += "Sequential([\n\t"
            char_node_ids.extend([orig_node_ids[node.value[0]]] * 14)
            for i, nde in enumerate(node.value):
                if i == 0:
                    string += str(nde)
                    char_node_ids.extend([orig_node_ids[nde]] * len(str(nde)))
                else:
                    string += ",\n\t" + str(nde)
                    char_node_ids.extend([orig_node_ids[nde]] * (len(str(nde)) + 3))
            string += "\n])"
            char_node_ids.extend([orig_node_ids[node.value[-1]]] * 3)
        else:
            string += str(node.value[0])
            char_node_ids.extend([orig_node_ids[node.value[0]]] * len(str(node.value[0])))
        if len(node.incoming_edges) > 0:
            substr = "(" + ",".join([f"value{node_id_map[x.source, x.value.index]}" for x in node.incoming_edges]) + ")"
            string += substr
            char_node_ids.extend([orig_node_ids[node.value[-1]]] * len(substr))
        string += "\n"
        char_node_ids.append(orig_node_ids[node.value[-1]])
    if return_node_ids:
        assert len(char_node_ids) == len(string)  # noqa: S101
        return string, char_node_ids
    return string


'''def _node_attrs_to_str(node: ONNXNode) -> str:
    match node.operation:
        case Operation.TRANSPOSE:
            if len(node.attributes["perm"]) == 1:
                return f"perm={node.attributes['perm'][0]}"
            return f"perm=[{str(node.attributes['perm']).replace(' ', '')[1:-1]}]"
        case Operation.MEAN:
            if len(node.attributes["axes"]) == 1:
                return f"axes={node.attributes['axes'][0]}"
            return f"axes=[{str(node.attributes['axes']).replace(' ', '')[1:-1]}]"
        case _:
            return ""'''


'''def encode_graph_old(graph: DAG[ONNXNode, ONNXEdge]) -> str:  # noqa: C901, PLR0912, PLR0915
    """Encode a DAG of ONNX nodes and edges into a string representation.

    Args:
        graph: The DAG to encode.

    Returns:
        The string representation of the encoded DAG.

    """
    non_io_nodes = [x for x in graph.nodes if x.value.index != -1]
    prev_node = None
    lines: list[str] = []
    curr_line: list[str] = []
    next_val_idx = 1
    node_output_name_map: dict[str, str] = {}

    # Iterate in ONNX-order over nodes
    for node in sorted(non_io_nodes, key=lambda x: x.value.index):
        incoming = [edge for edge in node.incoming_edges if edge.source.value.index != -1]
        outgoing = {edge.value.name for edge in node.outgoing_edges}
        # Multiple in- or outputs --> Singular node
        if len(incoming) != 1 or len(outgoing) != 1:
            # Complete previous chain, if there is one
            if prev_node is not None and len(curr_line) != 0:
                curr_line.append("")
                tmp = []
                for edge in prev_node.outgoing_edges:
                    if edge.value.name not in {x.value.name for x in tmp}:
                        tmp.append(edge)
                for i, edge in enumerate(tmp):
                    curr_line[-1] += f"Value{next_val_idx}"
                    if i == len(tmp) - 1:
                        curr_line[-1] += f":{'x'.join(map(str, edge.value.shape))}"
                    else:
                        curr_line[-1] += ", "
                    node_output_name_map[edge.value.name] = f"Value{next_val_idx}"
                    next_val_idx += 1
                lines.append(" --> ".join(curr_line))
            name = node.value.operation
            # Build argument list (first other nodes, then inputs and parameters)
            args = [
                node_output_name_map.get(edge.value.name, "prev")
                for edge in node.incoming_edges
                if edge.source.value.index != -1
            ]
            for edge in node.incoming_edges:
                if edge.source.value.index != -1:
                    continue
                if edge.source.value.operation == Operation.PARAM:
                    args.append("Param[" + ",".join(map(str, edge.value.shape)) + "]")
                else:
                    args.append("x".join(map(str, edge.value.shape)))
            attrs = _node_attrs_to_str(node.value)
            # Add this node and end the chain
            curr_line = [f"{name}({', '.join(args)})" + (f"({attrs})" if len(attrs) > 0 else ""), ""]
            tmp = []
            for edge in node.outgoing_edges:
                if edge.value.name not in {x.value.name for x in tmp}:
                    tmp.append(edge)
            for i, edge in enumerate(tmp):
                curr_line[-1] += f"Value{next_val_idx}"
                if i == len(tmp) - 1:
                    curr_line[-1] += f":{'x'.join(map(str, edge.value.shape))}"
                else:
                    curr_line[-1] += ", "
                node_output_name_map[edge.value.name] = f"Value{next_val_idx}"
                next_val_idx += 1
            lines.append(" --> ".join(curr_line))
            curr_line = []
        # Node doesnt take previous node as input --> Previous chain ends
        elif next(iter(incoming)).source != prev_node:
            # Complete previous chain, if there is one
            if prev_node is not None and len(curr_line) != 0:
                curr_line.append("")
                tmp = []
                for edge in prev_node.outgoing_edges:
                    if edge.value.name not in {x.value.name for x in tmp}:
                        tmp.append(edge)
                for i, edge in enumerate(tmp):
                    curr_line[-1] += f"Value{next_val_idx}"
                    if i == len(tmp) - 1:
                        curr_line[-1] += f":{'x'.join(map(str, edge.value.shape))}"
                    else:
                        curr_line[-1] += ", "
                    node_output_name_map[edge.value.name] = f"Value{next_val_idx}"
                    next_val_idx += 1
                lines.append(" --> ".join(curr_line))
            name = node.value.operation
            # Build argument list (first other nodes, then inputs and parameters)
            args = [
                node_output_name_map.get(edge.value.name, "prev")
                for edge in node.incoming_edges
                if edge.source.value.index != -1
            ]
            for edge in node.incoming_edges:
                if edge.source.value.index != -1:
                    continue
                if edge.source.value.operation == Operation.PARAM:
                    args.append("Param[" + ",".join(map(str, edge.value.shape)) + "]")
                else:
                    args.append("x".join(map(str, edge.value.shape)))
            # Add this node to the current chain
            attrs = _node_attrs_to_str(node.value)
            curr_line = [f"{name}({', '.join(args)})" + (f"({attrs})" if len(attrs) > 0 else "")]
        # Node can be added to current chain
        else:
            # Build argument list (first other nodes, then inputs and parameters)
            if len(curr_line) == 0:
                # Build arguments for chain start
                args = [
                    node_output_name_map.get(edge.value.name, "prev")
                    for edge in node.incoming_edges
                    if edge.source.value.index != -1
                ]
            else:
                # Trivial case: Chain center, 'prev' is the only input node
                args = ["prev"]
            for edge in node.incoming_edges:
                if edge.source.value.index != -1:
                    continue
                if edge.source.value.operation == Operation.PARAM:
                    args.append("Param[" + ",".join(map(str, edge.value.shape)) + "]")
                else:
                    args.append("x".join(map(str, edge.value.shape)))
            name = node.value.operation
            # Add this node to the current chain
            attrs = _node_attrs_to_str(node.value)
            curr_line.append(f"{name}({', '.join(args)})" + (f"({attrs})" if len(attrs) > 0 else ""))
        prev_node = node
    curr_line.append("Out")
    lines.append(" --> ".join(curr_line))

    return "\n".join(lines)'''


def parse_attribute(  # noqa: PLR0911
    attr: onnx.AttributeProto,
) -> (
    float
    | int
    | str
    | onnx.TensorProto
    | onnx.GraphProto
    | tuple[float, ...]
    | tuple[int, ...]
    | tuple[str, ...]
    | None
):
    """Parse an ONNX attribute to its corresponding Python value.

    Args:
        attr: The ONNX attribute to parse.

    Raises:
        ValueError: If the attribute type is not recognized.

    Returns:
        The parsed attribute value.
    """
    match attr.type:
        case onnx.AttributeProto.FLOAT:
            return attr.f
        case onnx.AttributeProto.INT:
            return attr.i
        case onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
        case onnx.AttributeProto.TENSOR:
            return attr.t
        case onnx.AttributeProto.FLOATS:
            return tuple(attr.floats)
        case onnx.AttributeProto.INTS:
            return tuple(attr.ints)
        case onnx.AttributeProto.STRINGS:
            return tuple(s.decode("utf-8") for s in attr.strings)
        case _:
            raise ValueError


def onnx_to_graph(  # noqa: C901, PLR0912, PLR0915
    onnx: onnx.ModelProto,
    *,
    optimize: bool = True,
    ignore_multiple_inputs: bool = False,
) -> DAG[ONNXNode, ONNXEdge]:
    """Convert an ONNX model to a DAG representation.

    Args:
        onnx: The ONNX model to convert.
        optimize: Whether to optimize the graph.
        ignore_multiple_inputs: Whether to ignore multiple input nodes instead of raising an error.

    Raises:
        ValueError: If multiple input nodes are found.

    Returns:
        The converted DAG representation of the ONNX model.
    """
    result = DAG[ONNXNode, ONNXEdge]()
    name_node_map: dict[str, Node[ONNXNode, ONNXEdge]] = {}
    shape_map: dict[str, tuple[int, ...]] = {
        x.name: tuple(y.dim_value for y in x.type.tensor_type.shape.dim) for x in onnx.graph.value_info
    }

    has_input = False
    # Load input nodes
    for onnx_tensor in onnx.graph.input:
        if onnx_tensor.name in [init.name for init in onnx.graph.initializer]:
            name_node_map[onnx_tensor.name] = result.add_node(
                ONNXNode(
                    name=onnx_tensor.name,
                    index=-1,
                    operation=Operation("Parameter"),
                    attributes={"shape": tuple(y.dim_value for y in onnx_tensor.type.tensor_type.shape.dim)},
                ),
            )
            shape_map[onnx_tensor.name] = name_node_map[onnx_tensor.name].value.attributes["shape"]
        else:
            if has_input and not ignore_multiple_inputs:
                msg = "Multiple input nodes found."
                raise ValueError(msg)
            has_input = True
            name_node_map[onnx_tensor.name] = result.add_node(
                ONNXNode(
                    name=onnx_tensor.name,
                    index=-1,
                    operation=Operation("Input"),
                    attributes={"shape": tuple(y.dim_value for y in onnx_tensor.type.tensor_type.shape.dim)},
                ),
            )
            shape_map[onnx_tensor.name] = name_node_map[onnx_tensor.name].value.attributes["shape"]
    # Differentiate parameter nodes from "normal" input nodes
    for onnx_tensor in onnx.graph.initializer:
        name_node_map[onnx_tensor.name] = result.add_node(
            ONNXNode(
                name=onnx_tensor.name,
                index=-1,
                operation=Operation("Parameter"),
                attributes={"shape": tuple(y for y in onnx_tensor.dims)},
            ),
        )
        shape_map[onnx_tensor.name] = name_node_map[onnx_tensor.name].value.attributes["shape"]
    # Load intermediate nodes
    for onnx_index, onnx_node in enumerate(onnx.graph.node):
        if onnx_node.op_type != Operation.CONST:
            name_node_map[onnx_node.name] = result.add_node(
                ONNXNode(
                    name=onnx_node.name,
                    index=onnx_index,
                    operation=Operation(onnx_node.op_type),
                    attributes={x.name: parse_attribute(x) for x in onnx_node.attribute},
                ),
            )
        else:
            name_node_map[onnx_node.name] = result.add_node(
                ONNXNode(
                    name=onnx_node.name,
                    index=onnx_index,
                    operation=Operation.CONST,
                    attributes={
                        "value": parse_attribute(next(x for x in onnx_node.attribute if x.name == "value")),
                        "shape": shape_map[onnx_node.output[0]],
                    },
                ),
            )
    # Load output nodes
    for onnx_tensor in onnx.graph.output:
        name_node_map[onnx_tensor.name] = result.add_node(
            ONNXNode(
                name=onnx_tensor.name,
                index=-1,
                operation=Operation.OUTPUT,
                attributes={"shape": tuple(y.dim_value for y in onnx_tensor.type.tensor_type.shape.dim)},
            ),
        )
        shape_map[onnx_tensor.name] = name_node_map[onnx_tensor.name].value.attributes["shape"]
    # Load edges between intermediate nodes
    for onnx_node1 in onnx.graph.node:
        for onnx_node2 in onnx.graph.node:
            common_tensors = set(onnx_node1.output) & set(onnx_node2.input)
            for name in onnx_node2.input:
                if name not in common_tensors:
                    continue
                _ = result.add_edge(
                    start=name_node_map[onnx_node1.name],
                    end=name_node_map[onnx_node2.name],
                    value=ONNXEdge(name=name, in_index=list(onnx_node2.input).index(name), shape=shape_map.get(name)),
                    check=False,
                )
    # Load edges between input/parameter nodes and intermediate nodes
    for onnx_node in onnx.graph.node:
        for onnx_tensor in chain(onnx.graph.input, onnx.graph.initializer):
            for name in onnx_node.input:
                if name != onnx_tensor.name:
                    continue
                _ = result.add_edge(
                    start=name_node_map[onnx_tensor.name],
                    end=name_node_map[onnx_node.name],
                    value=ONNXEdge(
                        name=onnx_tensor.name,
                        in_index=list(onnx_node.input).index(onnx_tensor.name),
                        shape=shape_map[onnx_tensor.name],
                    ),
                    check=False,
                )
    # Load edges between intermediate nodes and output nodes
    for onnx_node in onnx.graph.node:
        for onnx_tensor in onnx.graph.output:
            for name in onnx_node.output:
                if name != onnx_tensor.name:
                    continue
                _ = result.add_edge(
                    start=name_node_map[onnx_node.name],
                    end=name_node_map[onnx_tensor.name],
                    value=ONNXEdge(
                        name=onnx_tensor.name,
                        in_index=len(name_node_map[onnx_tensor.name].incoming_edges),
                        shape=shape_map[onnx_tensor.name],
                    ),
                    check=False,
                )

    # Try to infer in/out channels for convolution nodes
    for node in result.nodes:
        if node.value.operation == Operation.CONV:
            incoming_edges = sorted(node.incoming_edges, key=lambda e: e.value.index)
            node.value.attributes["in_channels"] = incoming_edges[1].source.value.attributes["shape"][
                1
            ] * node.value.attributes.get("group", 1)
            node.value.attributes["out_channels"] = incoming_edges[1].source.value.attributes["shape"][0]

    # Graph optimizations
    while True and optimize:
        node = None
        for node in result.nodes:
            incoming_edges = sorted(node.incoming_edges, key=lambda e: e.value.index)
            # Remove identity nodes
            if node.value.operation == Operation.IDENTITY:
                result.collapse_edge(incoming_edges[0], lambda x, _: x, check=False)
                break
            # Fuse convolutions with parameter nodes
            if (
                node.value.operation == Operation.CONV
                and len(node.incoming_edges) > 1
                and len([x for x in incoming_edges[1:] if x.source.value.operation == Operation.INPUT])
                == len(node.incoming_edges) - 1
            ):
                for i in range(1, len(node.incoming_edges)):
                    result.remove_edges(incoming_edges[i])
                break
            # Fuse norms with parameter nodes
            if (
                node.value.operation in {Operation.BATCHNORM, Operation.LAYERNORM, Operation.INSTANCENORM}
                and len(node.incoming_edges) > 1
                and len([x for x in incoming_edges[1:] if x.source.value.operation == Operation.INPUT])
                == len(node.incoming_edges) - 1
            ):
                for i in range(1, len(node.incoming_edges)):
                    result.remove_edges(incoming_edges[i])
                break
            # Fuse activations into convolutions
            if (
                node.value.operation in {Operation.RELU, Operation.HARD_SWISH}
                and incoming_edges[0].source.value.operation == Operation.CONV
            ):
                result.collapse_edge(incoming_edges[0], lambda x, _: x, check=False)
                node.value.attributes["activation"] = OP_ABBR_MAP[node.value.operation]
                break
            # Fuse additions with constant nodes
            if (
                node.value.operation == Operation.ADD
                and len([x for x in incoming_edges if x.source.value.operation == Operation.CONST]) > 0
            ):
                tmp = next(x for x in incoming_edges if x.source.value.operation == Operation.CONST)
                if "value" in node.value.attributes and type(node.value.attributes["value"]) is not type(
                    tmp.source.value.attributes["value"],
                ):
                    continue
                node.value.operation = Operation.ADD_CONST
                if "value" not in node.value.attributes:
                    node.value.attributes["value"] = tmp.source.value.attributes["value"]
                else:
                    node.value.attributes["value"] += tmp.source.value.attributes["value"]
                result.collapse_edge(tmp, lambda _, x: x, check=False)
                break
            # Fuse GEMM with parameter nodes into Linear
            if (
                node.value.operation == Operation.GEMM
                and len(node.incoming_edges) > 1
                and len([x for x in node.incoming_edges if x.source.value.operation == Operation.INPUT])
                == len(node.incoming_edges) - 1
            ):
                node.value.operation = Operation.LINEAR
                for edge in node.incoming_edges:
                    if edge.source.value.operation != Operation.INPUT:
                        continue
                    if len([x for x in edge.source.value.attributes["shape"] if x != 1]) == 2:  # noqa: PLR2004
                        node.value.attributes["in_features"] = edge.source.value.attributes["shape"][1]
                        node.value.attributes["out_features"] = edge.source.value.attributes["shape"][0]
                    result.remove_edges(edge)
                break
            # Remove dangling nodes
            if len(node.outgoing_edges) == 0 and node.value.operation != Operation.OUTPUT:
                result.remove_nodes(node)
                break
            # Remove unreachable nodes
            if len(node.incoming_edges) == 0 and node.value.operation not in {
                Operation.INPUT,
                Operation.PARAM,
                Operation.CONST,
            }:
                result.remove_nodes(node)
                break
        else:
            break
    return result
