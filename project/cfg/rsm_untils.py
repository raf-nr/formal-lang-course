from pyformlang.rsa import RecursiveAutomaton
from pyformlang.cfg import CFG


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """
    Converts a context-free grammar (CFG) into a recursive automaton (RSM).

    Args:
        cfg (CFG): The context-free grammar to convert.

    Returns:
        RecursiveAutomaton: The resulting recursive automaton equivalent to the provided CFG.
    """
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """
    Converts a string representing EBNF into a recursive automaton (RSM).

    Args:
        ebnf (str): The EBNF representation of the language.

    Returns:
        RecursiveAutomaton: The resulting recursive automaton equivalent to the provided EBNF.
    """
    return RecursiveAutomaton.from_text(ebnf)
