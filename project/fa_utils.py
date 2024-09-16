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
