import pytest
import networkx as nx
from project.fa_utils import regex_to_dfa, graph_to_nfa
from pyformlang.finite_automaton import Symbol, State


@pytest.mark.parametrize(
    "regex, accepted_strings, rejected_strings",
    [
        ("", [], [["a"], [""]]),
        ("a", [["a"]], [["b"], [""]]),
        ("ab|с", [["ab"], ["с"]], [["a"], ["b", "c"], ["abc"]]),
    ],
)
def test_regex_to_dfa(regex, accepted_strings, rejected_strings):
    dfa = regex_to_dfa(regex)

    assert dfa.is_deterministic()

    for string in accepted_strings:
        assert dfa.accepts([Symbol(s) for s in string])

    for string in rejected_strings:
        assert not dfa.accepts([Symbol(s) for s in string])


def test_graph_to_nfa_simple():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    nfa = graph_to_nfa(graph)

    assert State(0) in nfa.start_states
    assert State(1) in nfa.start_states
    assert State(2) in nfa.start_states

    assert State(0) in nfa.final_states
    assert State(1) in nfa.final_states
    assert State(2) in nfa.final_states

    assert nfa.accepts([Symbol("a"), Symbol("b")])
    assert not nfa.accepts([Symbol("a"), Symbol("a")])


def test_graph_to_nfa_with_custom_start_and_final_states():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    start_states = {0}
    final_states = {2}

    nfa = graph_to_nfa(graph, start_states=start_states, final_states=final_states)

    assert State(0) in nfa.start_states
    assert State(1) not in nfa.start_states
    assert State(2) not in nfa.start_states

    assert State(2) in nfa.final_states
    assert State(0) not in nfa.final_states
    assert State(1) not in nfa.final_states

    assert nfa.accepts([Symbol("a"), Symbol("b")])
    assert not nfa.accepts([Symbol("a")])


def test_graph_to_nfa_no_edges():
    graph = nx.MultiDiGraph()
    graph.add_node(0)
    graph.add_node(1)

    nfa = graph_to_nfa(graph)

    assert State(0) in nfa.start_states
    assert State(1) in nfa.start_states
    assert State(0) in nfa.final_states
    assert State(1) in nfa.final_states

    assert not nfa.accepts([Symbol("a")])


def test_graph_to_nfa_self_loop():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 0, label="a")

    nfa = graph_to_nfa(graph)

    assert State(0) in nfa.start_states

    assert State(0) in nfa.final_states

    assert nfa.accepts([Symbol("a")])
    assert not nfa.accepts([Symbol("b")])
