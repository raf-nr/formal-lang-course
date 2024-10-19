import networkx as nx

from pyformlang.cfg import CFG, Terminal

from project.cfg.normal_form import cfg_to_weak_normal_form


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """
    Perform a context-free path query (CFPQ) using the Hellings algorithm.

    Args:
        cfg (CFG): Context-free grammar defining the language.
        graph (nx.DiGraph): Directed graph to search for paths.
        start_nodes (set[int], optional): Nodes to start the path search from. Defaults to None, which considers all nodes.
        final_nodes (set[int], optional): Nodes to end the path search. Defaults to None, which considers all nodes.

    Returns:
        set[tuple[int, int]]: Pairs of nodes (start, end) representing paths that
        match the start symbol of the grammar.
    """
    weak_normal_form = cfg_to_weak_normal_form(cfg)

    cfpq_results = set()

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            for production in weak_normal_form.productions:
                if len(production.body) == 1 and isinstance(
                    production.body[0], Terminal
                ):
                    terminal = production.body[0].value
                    if terminal == label:
                        cfpq_results.add((u, production.head, v))

    nullable = weak_normal_form.get_nullable_symbols()
    for node in graph.nodes:
        for var in nullable:
            cfpq_results.add((node, var, node))

    added = True
    while added:
        added = False
        new_results = set()

        for v1, B, v2 in cfpq_results:
            for v2_, C, v3 in cfpq_results:
                if v2 == v2_:
                    for production in weak_normal_form.productions:
                        if (
                            len(production.body) == 2
                            and production.body[0] == B
                            and production.body[1] == C
                        ):
                            new_triple = (v1, production.head, v3)
                            if new_triple not in cfpq_results:
                                new_results.add(new_triple)
                                added = True

        cfpq_results.update(new_results)

    result_pairs = set()
    for u, var, v in cfpq_results:
        if var == weak_normal_form.start_symbol:
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result_pairs.add((u, v))

    return result_pairs
