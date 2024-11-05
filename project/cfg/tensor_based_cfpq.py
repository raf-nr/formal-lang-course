import networkx as nx
import scipy as sp

from pyformlang.rsa import RecursiveAutomaton

from project.fa import AdjacencyMatrixFA, intersect_automata
from project.fa_utils import graph_to_nfa, rsm_to_nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """
    Perform a context-free path query (CFPQ) using the Tensor based algorithm.

    Args:
        rsm (CFG): Recursive automaton defining the language.
        graph (nx.DiGraph): Directed graph to search for paths.
        start_nodes (set[int], optional): Nodes to start the path search from. Defaults to None, which considers all nodes.
        final_nodes (set[int], optional): Nodes to end the path search. Defaults to None, which considers all nodes.

    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) representing paths that
        match the start symbol of the recursive automaton (grammar).
    """
    graph = nx.MultiDiGraph(graph)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_adj_matrix = AdjacencyMatrixFA(graph_nfa)

    rsm_nfa = rsm_to_nfa(rsm)
    rsm_adj_matrix = AdjacencyMatrixFA(rsm_nfa)

    for nonterm in rsm.boxes:
        if graph_adj_matrix.boolean_decomposition.get(nonterm) is None:
            dimension = (graph_adj_matrix.states_amount, graph_adj_matrix.states_amount)
            graph_adj_matrix.boolean_decomposition[nonterm] = sp.sparse.csc_matrix(
                dimension, dtype=bool
            )

        if rsm_adj_matrix.boolean_decomposition.get(nonterm) is None:
            dimension = (rsm_adj_matrix.states_amount, rsm_adj_matrix.states_amount)
            rsm_adj_matrix.boolean_decomposition[nonterm] = sp.sparse.csc_matrix(
                dimension, dtype=bool
            )

    prev_nonzero_count = -1
    new_nonzero_count = 0

    while prev_nonzero_count != new_nonzero_count:
        prev_nonzero_count = new_nonzero_count

        automata_intersection = intersect_automata(rsm_adj_matrix, graph_adj_matrix)
        reachability_matrix = automata_intersection.transitive_closure()
        state_mapping = {
            idx: state for state, idx in automata_intersection.states_indices.items()
        }

        src_indices, dest_indices = reachability_matrix.nonzero()

        for idx in range(len(src_indices)):
            src_idx = src_indices[idx]
            dest_idx = dest_indices[idx]

            src_inner_rsm_state, src_graph_node = state_mapping[src_idx].value
            src_symbol, src_rsm_node = src_inner_rsm_state.value
            dest_inner_rsm_state, dest_graph_node = state_mapping[dest_idx].value
            dest_symbol, dest_rsm_node = dest_inner_rsm_state.value

            if src_symbol != dest_symbol:
                continue

            src_rsm_states = rsm.boxes[src_symbol].dfa.start_states
            dest_rsm_states = rsm.boxes[src_symbol].dfa.final_states

            if src_rsm_node in src_rsm_states and dest_rsm_node in dest_rsm_states:
                graph_adj_matrix.boolean_decomposition[src_symbol][
                    graph_adj_matrix.states_indices[src_graph_node],
                    graph_adj_matrix.states_indices[dest_graph_node],
                ] = True

        new_nonzero_count = 0
        for symbol, adj_matrix in graph_adj_matrix.boolean_decomposition.items():
            nonzero_for_symbol = adj_matrix.count_nonzero()
            new_nonzero_count += nonzero_for_symbol

    cfpq_result = set()

    for start_state in graph_adj_matrix.start_states:
        for final_state in graph_adj_matrix.final_states:
            if graph_adj_matrix.boolean_decomposition[rsm.initial_label][
                graph_adj_matrix.states_indices[start_state],
                graph_adj_matrix.states_indices[final_state],
            ]:
                cfpq_result.add((start_state.value, final_state.value))

    return cfpq_result
