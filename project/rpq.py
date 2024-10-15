from networkx import MultiDiGraph

from project.fa_ import AdjacencyMatrixFA, intersect_automata
from project.fa_utils import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
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
