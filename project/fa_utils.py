import networkx as nx
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)
from pyformlang.regular_expression import Regex
from pyformlang.rsa import RecursiveAutomaton


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """
    Converts a regular expression to a minimized deterministic finite automaton (DFA).

    Args:
        regex (str): The regular expression to convert.

    Returns:
        DeterministicFiniteAutomaton: The minimized DFA equivalent of the regular expression.
    """
    regex_obj = Regex(regex)
    nfa = regex_obj.to_epsilon_nfa()
    dfa = nfa.to_deterministic().minimize()
    return dfa


def graph_to_nfa(
    graph: nx.MultiDiGraph,
    start_states: set[int] | None = None,
    final_states: set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    """
    Converts a directed labeled graph (MultiDiGraph) to a nondeterministic finite automaton (NFA).

    Args:
        graph (nx.MultiDiGraph): The input graph.
        start_states (Optional[Set[int]]): The set of start states. If None, all nodes are considered start states.
        final_states (Optional[Set[int]]): The set of final states. If None, all nodes are considered final states.

    Returns:
        NondeterministicFiniteAutomaton: The resulting NFA.
    """
    nfa = NondeterministicFiniteAutomaton()

    start_states = start_states or set(graph.nodes)
    final_states = final_states or set(graph.nodes)

    for u, v, data in graph.edges(data=True):
        symbol = Symbol(data["label"])
        nfa.add_transition(State(u), symbol, State(v))

    for state in start_states:
        nfa.add_start_state(State(state))
    for state in final_states:
        nfa.add_final_state(State(state))

    return nfa


def rsm_to_nfa(automaton: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    """
    Converts a recursive automaton (RSM) to a nondeterministic finite automaton (NFA).

    Args:
        automaton (RecursiveAutomaton): The recursive automaton to convert.

    Returns:
        NondeterministicFiniteAutomaton: The resulting NFA equivalent to the recursive automaton.
    """
    result_nfa = NondeterministicFiniteAutomaton()

    for rule, container in automaton.boxes.items():
        deterministic_automaton = container.dfa

        start_end_states = deterministic_automaton.start_states.union(
            deterministic_automaton.final_states
        )
        for state in start_end_states:
            combined_state = State((rule, state))
            if state in deterministic_automaton.final_states:
                result_nfa.add_final_state(combined_state)
            if state in deterministic_automaton.start_states:
                result_nfa.add_start_state(combined_state)

        transitions = deterministic_automaton.to_networkx().edges(data="label")
        for origin, destination, transition_label in transitions:
            initial_state = State((rule, origin))
            target_state = State((rule, destination))
            result_nfa.add_transition(initial_state, transition_label, target_state)

    return result_nfa
