from dataclasses import dataclass
from enum import StrEnum, auto

import networkx as nx

from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import Symbol, DeterministicFiniteAutomaton


@dataclass(frozen=True)
class RSMState:
    variable: Symbol
    substate: str


class RSMStateFinalFlag(StrEnum):
    final = auto()
    non_final = auto()


@dataclass(frozen=True)
class RSMStateInfo:
    terminal_edges: dict[Symbol, RSMState]
    variable_edges: dict[Symbol, tuple[RSMState, RSMState]]
    is_final: RSMStateFinalFlag


@dataclass(frozen=True)
class SPPFNode:
    gss_node: "GSSNode"
    state: RSMState
    node: int


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """
    Perform a context-free path query (CFPQ) using the GLL based algorithm.

    Args:
        rsm (CFG): Recursive automaton defining the language.
        graph (nx.DiGraph): Directed graph to search for paths.
        start_nodes (set[int], optional): Nodes to start the path search from. Defaults to None, which considers all nodes.
        final_nodes (set[int], optional): Nodes to end the path search. Defaults to None, which considers all nodes.

    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) representing paths that
        match the start symbol of the recursive automaton (grammar).
    """
    reachable_pairs = set()
    start_nodes = start_nodes if start_nodes else set(graph.nodes)
    final_nodes = final_nodes if final_nodes else set(graph.nodes)

    engine = GllBasedCFPQEngine(rsm, graph)

    for node in start_nodes:
        gss_node = engine.stack_manager.pieck(engine.entry_point, node)
        gss_node.push(RSMState(Symbol("$"), "fin"), engine.final_node)
        nodes = {SPPFNode(gss_node, engine.entry_point, node)}
        nodes.difference_update(engine.tracked_nodes)
        engine.tracked_nodes.update(nodes)
        engine.to_process.update(nodes)

    while engine.to_process:
        reachable_pairs.update(engine.execute_step(engine.to_process.pop()))

    return {
        (start, end)
        for start, end in reachable_pairs
        if start in start_nodes and end in final_nodes
    }


class GSS:
    def __init__(self):
        self.figure: dict[tuple[RSMState, int], "GSSNode"] = {}

    def pieck(self, rsm_state: RSMState, node: int):
        node_instance = self.figure.get((rsm_state, node))
        if node_instance is None:
            node_instance = GSSNode(node)
            self.figure[(rsm_state, node)] = node_instance
        return node_instance


class GSSNode:
    def __init__(self, identifier: int):
        self.identifier = identifier
        self.arcs = {}
        self.visited_checkpoints = set()

    def push(self, state: RSMState, target_node: "GSSNode") -> set[SPPFNode]:
        new_nodes = set()
        if state not in self.arcs:
            self.arcs[state] = {target_node}
        else:
            if target_node not in self.arcs[state]:
                self.arcs[state].add(target_node)

        for checkpoint in self.visited_checkpoints:
            new_nodes.add(SPPFNode(target_node, state, checkpoint))
        return new_nodes

    def pop(self, checkpoint: int) -> set[SPPFNode]:
        parsed_nodes = set()
        if checkpoint in self.visited_checkpoints:
            return parsed_nodes

        self.visited_checkpoints.add(checkpoint)

        for state, linked_nodes in self.arcs.items():
            parsed_nodes.update(
                SPPFNode(linked_node, state, checkpoint) for linked_node in linked_nodes
            )

        return parsed_nodes


class GllBasedCFPQEngine:
    def __init__(self, rsm: RecursiveAutomaton, graph: nx.DiGraph):
        self.node_links: dict[int, dict[Symbol, set[int]]] = {}
        self.rsm_states: dict[Symbol, dict[str, RSMStateInfo]] = {}
        self.entry_point: RSMState

        self.stack_manager = GSS()
        self.final_node = self.stack_manager.pieck(RSMState(Symbol("$"), "halt"), -1)

        self.to_process: set[SPPFNode] = set()
        self.tracked_nodes: set[SPPFNode] = set()

        for vertex in graph.nodes():
            self.node_links[vertex] = {}

        for src, dest, label in graph.edges(data="label"):
            if label is not None:
                transitions = self.node_links[src]
                destinations = transitions[label] if label in transitions else set()
                destinations.add(dest)
                transitions[label] = destinations

        for key in rsm.boxes:
            self.rsm_states[key] = {}

        for key, automaton_box in rsm.boxes.items():
            automaton: DeterministicFiniteAutomaton = automaton_box.dfa
            graph_representation = automaton.to_networkx()
            state_map = self.rsm_states[key]

            for node in graph_representation.nodes:
                is_terminal = node in automaton.final_states
                is_final = (
                    RSMStateFinalFlag.final
                    if is_terminal
                    else RSMStateFinalFlag.non_final
                )
                state_map[node] = RSMStateInfo({}, {}, is_final)

            for origin, target, edge_label in graph_representation.edges(data="label"):
                if edge_label is not None:
                    transition_data = state_map[origin]
                    if Symbol(edge_label) not in self.rsm_states:
                        transition_data.terminal_edges[edge_label] = RSMState(
                            key, target
                        )
                    else:
                        nested_automaton: DeterministicFiniteAutomaton = rsm.boxes[
                            Symbol(edge_label)
                        ].dfa
                        start_state = nested_automaton.start_state.value
                        transition_data.variable_edges[edge_label] = (
                            RSMState(Symbol(edge_label), start_state),
                            RSMState(key, target),
                        )

        start_var = rsm.initial_label
        start_automaton: DeterministicFiniteAutomaton = rsm.boxes[start_var].dfa
        self.entry_point = RSMState(start_var, start_automaton.start_state.value)

    def execute_step(self, current_node: SPPFNode) -> set[tuple[int, int]]:
        def split_nodes(
            nodes: set[SPPFNode], previous_node: SPPFNode
        ) -> tuple[set[SPPFNode], set[tuple[int, int]]]:
            remaining_nodes, complete_pairs = set(), set()

            for node in nodes:
                if node.gss_node == self.final_node:
                    complete_pairs.add((previous_node.gss_node.identifier, node.node))
                else:
                    remaining_nodes.add(node)

            return remaining_nodes, complete_pairs

        def update_nodes(nodes: set[SPPFNode]):
            nodes.difference_update(self.tracked_nodes)
            self.tracked_nodes.update(nodes)
            self.to_process.update(nodes)

        node_state = current_node.state
        state_data = self.rsm_states[node_state.variable][node_state.substate]

        for terminal, target_state in state_data.terminal_edges.items():
            graph_transitions = self.node_links[current_node.node]
            if terminal in graph_transitions:
                new_nodes = {
                    SPPFNode(current_node.gss_node, target_state, target)
                    for target in graph_transitions[terminal]
                }
                update_nodes(new_nodes)

        completed_pairs = set()
        for variable, (entry_state, return_state) in state_data.variable_edges.items():
            intermediate_node = self.stack_manager.pieck(entry_state, current_node.node)
            pending_nodes = intermediate_node.push(return_state, current_node.gss_node)

            remaining, completed = split_nodes(pending_nodes, current_node)
            update_nodes(remaining)
            sppf_nodes = {SPPFNode(intermediate_node, entry_state, current_node.node)}
            update_nodes(sppf_nodes)
            completed_pairs.update(completed)

        if state_data.is_final == RSMStateFinalFlag.final:
            popped_nodes = current_node.gss_node.pop(current_node.node)
            remaining, completed = split_nodes(popped_nodes, current_node)
            update_nodes(remaining)
            completed_pairs.update(completed)

        return completed_pairs
