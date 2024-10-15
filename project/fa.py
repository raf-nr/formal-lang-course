import functools
import operator
import numpy as np
import scipy as sp

from dataclasses import dataclass
from typing import Iterable

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State


@dataclass
class AdjacencyMatrixFAData:
    """
    Data structure to represent a finite automaton using adjacency matrices.

    Attributes:
        states (set[State]): Set of all states in the automaton.
        start_states (set[State]): Set of initial states.
        final_states (set[State]): Set of accepting (final) states.
        states_indices (dict[State, int]): Mapping from states to indices.
        boolean_decomposition (dict[Symbol, sp.sparse.csc_matrix]): Boolean matrices representing transitions for each symbol.
    """

    states: set[State]
    start_states: set[State]
    final_states: set[State]
    states_indices: dict[State, int]
    boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix]


class AdjacencyMatrixFA:
    """
    A class to represent a finite automaton using adjacency matrices.

    Supports conversion of an NFA to an adjacency matrix form and provides
    methods for basic operations like checking acceptance and calculating
    transitive closure.
    """

    def __init__(
        self, automaton: NondeterministicFiniteAutomaton | AdjacencyMatrixFAData
    ):
        """
        Initializes the adjacency matrix representation of the automaton.

        Args:
            automaton (NondeterministicFiniteAutomaton | AdjacencyMatrixFAData):
                The automaton to convert or an already prepared data structure.
        """
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
        """
        Returns the boolean matrices representing transitions for each symbol.
        """
        return self._boolean_decomposition

    @property
    def states(self) -> set[State]:
        """Returns the set of all states in the automaton."""
        return self._states

    @property
    def start_states(self) -> set[State]:
        """Returns the set of initial states in the automaton."""
        return self._start_states

    @property
    def final_states(self) -> set[State]:
        """Returns the set of accepting (final) states in the automaton."""
        return self._final_states

    @property
    def states_amount(self) -> int:
        """Returns the number of states in the automaton."""
        return self._states_amount

    @property
    def states_indices(self) -> dict[State, int]:
        """Returns the mapping from states to indices."""
        return self._states_indices

    @property
    def start_states_indices(self) -> set[int]:
        """Returns the indices of initial states."""
        return self._start_states_indices

    @property
    def final_states_indices(self) -> set[int]:
        """Returns the indices of accepting (final) states."""
        return self._final_states_indices

    @property
    def start_configuration(self) -> np.ndarray:
        """Creates an array indicating the initial configuration of the automaton."""
        start_config = np.zeros(self._states_amount, dtype=bool)
        for start_state_index in self._start_states_indices:
            start_config[start_state_index] = True
        return start_config

    @property
    def final_configuration(self) -> np.ndarray:
        """Creates an array indicating the accepting configuration of the automaton."""
        final_config = np.zeros(self._states_amount, dtype=bool)
        for final_state_index in self._final_states_indices:
            final_config[final_state_index] = True
        return final_config

    def transitive_closure(self) -> sp.sparse.csc_matrix:
        """
        Computes the transitive closure (reachability matrix) of the automaton adjacency matrix.

        Returns:
            sp.sparse.csc_matrix: The transitive closure matrix.
        """
        matrices_list = list(self._boolean_decomposition.values())
        init_matrix = sp.sparse.csc_matrix(self._matrix_size, dtype=bool)
        main_diagonal_indices = np.arange(self._matrix_size[0])
        init_matrix[main_diagonal_indices, main_diagonal_indices] = True
        common_matrix = functools.reduce(operator.add, matrices_list, init_matrix)
        closure = common_matrix**self._states_amount
        return closure

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """
        Checks if the automaton accepts the given word.

        Args:
            word (Iterable[Symbol]): A sequence of symbols.

        Returns:
            bool: True if the automaton accepts the word, False otherwise.
        """
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
        """
        Checks if the automaton's language is empty.

        Returns:
            bool: True if the automaton accepts no words, False otherwise.
        """
        transitive_closure = self.transitive_closure()
        empty = True
        for start in self._start_states_indices:
            for final in self._final_states_indices:
                empty = False if transitive_closure[start, final] else empty
        return empty

    def print_matrices(self) -> None:
        """Prints the adjacency matrices for each symbol."""
        for symbol, matrix in self._boolean_decomposition.items():
            print(f"Matrix for symbol '{symbol}':")
            print(matrix.toarray())
            print()


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """
    Constructs a new automaton's adjacency matrix that is the intersection of two input automata.

    Args:
        automaton1 (AdjacencyMatrixFA): The first automaton adjacency matrix.
        automaton2 (AdjacencyMatrixFA): The second automaton adjacency matrix.

    Returns:
        AdjacencyMatrixFA: The resulting automaton adjacency matrix after intersection.
    """
    states: set[State] = set()
    start_states: set[State] = set()
    final_states: set[State] = set()
    states_indices: dict[State, int] = {}
    boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix] = {}

    for first_automaton_state in automaton1.states:
        for second_automaton_state in automaton2.states:
            first_automaton_state_index = automaton1.states_indices[
                first_automaton_state
            ]
            second_automaton_state_index = automaton2.states_indices[
                second_automaton_state
            ]

            new_state = State((first_automaton_state, second_automaton_state))
            new_automaton_state_index = (
                automaton2.states_amount * first_automaton_state_index
                + second_automaton_state_index
            )

            states.add(new_state)
            states_indices[new_state] = new_automaton_state_index

            if (
                first_automaton_state in automaton1.start_states
                and second_automaton_state in automaton2.start_states
            ):
                start_states.add(new_state)
            if (
                first_automaton_state in automaton1.final_states
                and second_automaton_state in automaton2.final_states
            ):
                final_states.add(new_state)

    for symbol in automaton1.boolean_decomposition:
        if symbol not in automaton2.boolean_decomposition:
            continue

        first_automaton_symbol_adj = automaton1.boolean_decomposition[symbol]
        second_automaton_symbol_adj = automaton2.boolean_decomposition[symbol]

        new_automaton_symbol_adj = sp.sparse.kron(
            first_automaton_symbol_adj, second_automaton_symbol_adj, format="csc"
        )
        boolean_decomposition[symbol] = new_automaton_symbol_adj

    return AdjacencyMatrixFA(
        AdjacencyMatrixFAData(
            states=states,
            start_states=start_states,
            final_states=final_states,
            states_indices=states_indices,
            boolean_decomposition=boolean_decomposition,
        )
    )
