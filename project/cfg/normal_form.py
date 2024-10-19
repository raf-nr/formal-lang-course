from pyformlang.cfg import CFG, Variable, Production, Epsilon


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """
    Convert a context-free grammar (CFG) to its weak normal form.

    Args:
        cfg (CFG): The input context-free grammar.

    Returns:
        CFG: A new context-free grammar in weak normal form.
    """
    normal_form_cfg = cfg.to_normal_form()
    nullable = cfg.get_nullable_symbols()

    new_productions = set(normal_form_cfg.productions)
    for var in nullable:
        new_productions.add(Production(Variable(var.value), [Epsilon()]))

    weak_normal_form_cfg = CFG(
        start_symbol=cfg.start_symbol, productions=new_productions
    )
    weak_normal_form_cfg = weak_normal_form_cfg.remove_useless_symbols()

    return weak_normal_form_cfg
