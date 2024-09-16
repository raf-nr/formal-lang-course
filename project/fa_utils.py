import networkx as nx
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)
from pyformlang.regular_expression import Regex


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
