from typing import Any

import networkx as nx

from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_matrix

from project.cfg.normal_form import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """
    Perform a context-free path query (CFPQ) using the Hellings algorithm with matrix-based approach.

    Args:
        cfg (CFG): Context-free grammar defining the language.
        graph (nx.DiGraph): Directed graph to search for paths.
        start_nodes (set[int], optional): Nodes to start the path search from. Defaults to None, which considers all nodes.
        final_nodes (set[int], optional): Nodes to end the path search. Defaults to None, which considers all nodes.

    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) representing paths that
        match the start symbol of the grammar.
    """
    weak_normal_form: CFG = cfg_to_weak_normal_form(cfg)
    nodes_amount: int = graph.number_of_nodes()
    node_to_index: dict[Any, int] = {
        node: idx for idx, node in enumerate(graph.nodes())
    }
    index_to_node: dict[int, Any] = {idx: node for node, idx in node_to_index.items()}

    var_matrices: dict[Variable, csr_matrix] = {}
    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            for production in weak_normal_form.productions:
                if len(production.body) == 1 and isinstance(
                    production.body[0], Terminal
                ):
                    terminal = production.body[0].value
                    if terminal == label:
                        head = production.head
                        if head not in var_matrices:
                            var_matrices[head] = csr_matrix(
                                (nodes_amount, nodes_amount), dtype=bool
                            )
                        var_matrices[head][node_to_index[u], node_to_index[v]] = True

    nullable = weak_normal_form.get_nullable_symbols()
    for node in graph.nodes:
        for var in nullable:
            var = Variable(var.value)
            if var not in var_matrices:
                var_matrices[var] = csr_matrix((nodes_amount, nodes_amount), dtype=bool)
            var_matrices[var][node_to_index[node], node_to_index[node]] = True

    added = True
    while added:
        added = False
        for production in weak_normal_form.productions:
            if len(production.body) == 2:
                B = Variable(production.body[0].value)
                C = Variable(production.body[1].value)
                head = production.head
                if B in var_matrices and C in var_matrices:
                    if head not in var_matrices:
                        var_matrices[head] = csr_matrix(
                            (nodes_amount, nodes_amount), dtype=bool
                        )

                    new_mat = var_matrices[B] @ var_matrices[C]
                    new_mat_coo = new_mat.tocoo()

                    for u, v, value in zip(
                        new_mat_coo.row, new_mat_coo.col, new_mat_coo.data
                    ):
                        if value and not var_matrices[head][u, v]:
                            var_matrices[head][u, v] = True
                            added = True

    result_pairs = set()
    start_symbol = weak_normal_form.start_symbol
    if start_symbol in var_matrices:
        final_matrix = var_matrices[start_symbol].tocoo()
        for u_idx, v_idx in zip(final_matrix.row, final_matrix.col):
            u = index_to_node[u_idx]
            v = index_to_node[v_idx]
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result_pairs.add((u, v))

    return result_pairs
