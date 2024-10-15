import scipy as sp

from dataclasses import dataclass

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State


@dataclass
class AdjacencyMatrixFAData:
    states: set[State]
    start_states: set[State]
    final_states: set[State]
    states_indices: dict[State, int]
    boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix]


class AdjacencyMatrixFA:
    def __init__(
        self, automaton: NondeterministicFiniteAutomaton | AdjacencyMatrixFAData
    ):
        self._states: set[State] = automaton.states
        self._start_states: set[State] = automaton.start_states
        self._final_states: set[State] = automaton.final_states
        self._states_amount: int = len(self._states)

        self._states_indices: dict[State, int] = (
            automaton.states_indices
            if isinstance(automaton, AdjacencyMatrixFAData)
            else {st: i for (i, st) in enumerate(automaton.states)}
        )
        self._start_states_indices: set[int] = set(
            self._states_indices[st] for st in automaton.start_states
        )
        self._final_states_indices: set[int] = set(
            self._states_indices[st] for st in automaton.final_states
        )

        self._matrix_size: tuple[int, int] = (self._states_amount, self._states_amount)
        self._boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix] = {}

        if isinstance(automaton, AdjacencyMatrixFAData):
            self._boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix] = (
                automaton.boolean_decomposition
            )
        else:
            graph = automaton.to_networkx()
            for u, v, label in graph.edges(data="label"):
                if label:
                    symbol = Symbol(label)
                    if symbol not in self._boolean_decomposition:
                        self._boolean_decomposition[symbol] = sp.sparse.csc_matrix(
                            self._matrix_size, dtype=bool
                        )
                    self._boolean_decomposition[symbol][
                        self._states_indices[u], self._states_indices[v]
                    ] = True
