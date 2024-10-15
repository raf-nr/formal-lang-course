import functools
import operator
import scipy as sp
import numpy as np

from dataclasses import dataclass
from typing import Iterable

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

    @property
    def boolean_decomposition(self) -> dict[Symbol, sp.sparse.csc_matrix]:
        return self._boolean_decomposition

    @property
    def states(self) -> set[State]:
        return self._states

    @property
    def start_states(self) -> set[State]:
        return self._start_states

    @property
    def final_states(self) -> set[State]:
        return self._final_states

    @property
    def states_amount(self) -> int:
        return self._states_amount

    @property
    def states_indices(self) -> dict[State, int]:
        return self._states_indices

    @property
    def start_states_indices(self) -> set[int]:
        return self._start_states_indices

    @property
    def final_states_indices(self) -> set[int]:
        return self._final_states_indices

    @property
    def start_configuration(self) -> np.ndarray:
        start_config = np.zeros(self._states_amount, dtype=bool)
        for start_state_index in self._start_states_indices:
            start_config[start_state_index] = True
        return start_config

    @property
    def final_configuration(self) -> np.ndarray:
        final_config = np.zeros(self._states_amount, dtype=bool)
        for final_state_index in self._final_states_indices:
            final_config[final_state_index] = True
        return final_config

    def transitive_closure(self) -> sp.sparse.csc_matrix:
        matrices_list = list(self._boolean_decomposition.values())
        init_matrix = sp.sparse.csc_matrix(self._matrix_size, dtype=bool)
        main_diagonal_indices = np.arange(self._matrix_size[0])
        init_matrix[main_diagonal_indices, main_diagonal_indices] = True
        common_matrix = functools.reduce(operator.add, matrices_list, init_matrix)
        closure = common_matrix**self._states_amount
        return closure

    def accepts(self, word: Iterable[Symbol]) -> bool:
        start_config = self.start_configuration
        final_config = self.final_configuration

        current_config = start_config.copy()

        for symbol in word:
            if symbol not in self._boolean_decomposition:
                return False
            matrix = self._boolean_decomposition[symbol]
            current_config = current_config @ matrix.toarray()
        return np.any(current_config & final_config)

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()
        empty = True
        for start in self._start_states_indices:
            for final in self._final_states_indices:
                empty = False if transitive_closure[start, final] else empty
        return empty

    def print_matrices(self) -> None:
        for symbol, matrix in self._boolean_decomposition.items():
            print(f"Matrix for symbol '{symbol}':")
            print(matrix.toarray())
            print()
