import scipy as sp

from networkx import MultiDiGraph

from project.fa import AdjacencyMatrixFA, intersect_automata
from project.fa_utils import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """
    Perform a Regular Path Query (RPQ) using a tensor-based approach.

    Args:
        regex (str): Regular expression defining the query (limitation).
        graph (MultiDiGraph): Directed graph to search for paths.
        start_nodes (set[int]): Nodes to start the path search from.
        final_nodes (set[int]): Nodes to end the path search.

    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) with paths matching the regex.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    graph_nfa_adj_matrix = AdjacencyMatrixFA(graph_nfa)
    regex_dfa_adj_matrix = AdjacencyMatrixFA(regex_dfa)

    adj_matrix_intersection = intersect_automata(
        graph_nfa_adj_matrix, regex_dfa_adj_matrix
    )
    reachability_matrix = adj_matrix_intersection.transitive_closure()

    result = set()

    inverted_states_indices_dict = {
        value: key for key, value in adj_matrix_intersection.states_indices.items()
    }

    for start in adj_matrix_intersection.start_states_indices:
        for final in adj_matrix_intersection.final_states_indices:
            if reachability_matrix[start, final]:
                graph_start_state = inverted_states_indices_dict[start].value[0]
                graph_final_state = inverted_states_indices_dict[final].value[0]
                result.add((graph_start_state, graph_final_state))

    return result


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """
    Perform a Regular Path Query (RPQ) using a multiple-source BFS based approach.

    Args:
        regex (str): Regular expression defining the query (limitation).
        graph (MultiDiGraph): Directed graph to search for paths.
        start_nodes (set[int]): Nodes to start the path search from.
        final_nodes (set[int]): Nodes to end the path search.

    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) with paths matching the regex.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    graph_nfa_adj_matrix = AdjacencyMatrixFA(graph_nfa)
    regex_dfa_adj_matrix = AdjacencyMatrixFA(regex_dfa)

    start_states_indices_pairs = [
        (dfa_start_state_index, nfa_start_state_index)
        for dfa_start_state_index in regex_dfa_adj_matrix.start_states_indices
        for nfa_start_state_index in graph_nfa_adj_matrix.start_states_indices
    ]
    alphabet = (
        graph_nfa_adj_matrix.boolean_decomposition.keys()
        & regex_dfa_adj_matrix.boolean_decomposition.keys()
    )
    inverted_nfa_states_indices_dict = {
        value: key for key, value in graph_nfa_adj_matrix.states_indices.items()
    }

    regex_dfa_states_amount = regex_dfa_adj_matrix.states_amount
    graph_nfa_states_amount = graph_nfa_adj_matrix.states_amount
    start_states_amount = len(start_states_indices_pairs)

    matrices = []
    for regex_dfa_state, graph_nfa_state in start_states_indices_pairs:
        matrix = sp.sparse.csc_matrix(
            (regex_dfa_states_amount, graph_nfa_states_amount), dtype=bool
        )
        matrix[regex_dfa_state, graph_nfa_state] = True
        matrices.append(matrix)

    initial_front: sp.sparse.csc_matrix = sp.sparse.vstack(
        matrices, format="csc", dtype=bool
    )
    front: sp.sparse.csc_matrix = initial_front
    visited: sp.sparse.csc_matrix = front

    while front.count_nonzero() > 0:
        fronts_for_each_symbol = []

        for symbol in alphabet:
            new_front = front @ graph_nfa_adj_matrix.boolean_decomposition[symbol]
            new_front_matrices = [
                regex_dfa_adj_matrix.boolean_decomposition[symbol].transpose()
                @ new_front[
                    regex_dfa_states_amount * i : regex_dfa_states_amount * (i + 1)
                ]
                for i in range(start_states_amount)
            ]
            new_front = sp.sparse.vstack(new_front_matrices, format="csc", dtype=bool)
            fronts_for_each_symbol.append(new_front)

        front = sum(fronts_for_each_symbol) > visited
        visited += front

    result = set()

    for i, (_, graph_nfa_start_state_index) in enumerate(start_states_indices_pairs):
        visited_block = visited[
            regex_dfa_states_amount * i : regex_dfa_states_amount * (i + 1)
        ]

        for regex_dfa_final_state_index in regex_dfa_adj_matrix.final_states_indices:
            for (
                graph_nfa_final_state_index
            ) in graph_nfa_adj_matrix.final_states_indices:
                if visited_block[
                    regex_dfa_final_state_index, graph_nfa_final_state_index
                ]:
                    result.add(
                        (
                            inverted_nfa_states_indices_dict[
                                graph_nfa_start_state_index
                            ].value,
                            inverted_nfa_states_indices_dict[
                                graph_nfa_final_state_index
                            ].value,
                        )
                    )

    return result
