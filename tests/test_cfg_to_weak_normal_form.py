import pytest

from pyformlang.cfg import Production, Variable, Epsilon, Terminal, CFG

from project.cfg.normal_form import cfg_to_weak_normal_form


@pytest.mark.parametrize(
    "productions, expected_productions",
    [
        # Test case 1: Simple grammar with nullable production
        (
            {
                Production(Variable("S"), [Variable("A")]),
                Production(Variable("A"), [Terminal("a")]),
                Production(Variable("A"), [Epsilon()]),
            },
            {
                Production(Variable("S"), [Terminal("a")]),
                Production(Variable("S"), [Epsilon()]),
            },
        ),
        # Test case 2: Grammar with multiple nullable variables
        (
            {
                Production(Variable("S"), [Variable("A"), Variable("B")]),
                Production(Variable("A"), [Terminal("a")]),
                Production(Variable("A"), [Epsilon()]),
                Production(Variable("B"), [Terminal("b")]),
                Production(Variable("B"), [Epsilon()]),
            },
            {
                Production(Variable("S"), [Variable("A"), Variable("B")]),
                Production(Variable("S"), [Terminal("a")]),
                Production(Variable("S"), [Terminal("b")]),
                Production(Variable("S"), [Epsilon()]),
                Production(Variable("A"), [Terminal("a")]),
                Production(Variable("B"), [Terminal("b")]),
                Production(Variable("A"), [Epsilon()]),
                Production(Variable("B"), [Epsilon()]),
            },
        ),
        # Test case 3: Grammar with no nullable variables
        (
            {
                Production(Variable("S"), [Variable("A")]),
                Production(Variable("A"), [Terminal("a")]),
                Production(Variable("B"), [Terminal("b")]),
            },
            {
                Production(Variable("S"), [Terminal("a")]),
            },
        ),
        # Test case 4: Grammar that is already in weak normal form
        (
            {
                Production(Variable("S"), [Terminal("a")]),
                Production(Variable("S"), [Epsilon()]),
            },
            {
                Production(Variable("S"), [Terminal("a")]),
                Production(Variable("S"), [Epsilon()]),
            },
        ),
        # Test case 5: Grammar with only epsilon productions
        (
            {
                Production(Variable("S"), [Epsilon()]),
                Production(Variable("A"), [Epsilon()]),
            },
            {
                Production(Variable("S"), [Epsilon()]),
            },
        ),
    ],
)
def test_cfg_to_weak_normal_form(productions, expected_productions):
    cfg = CFG(start_symbol=Variable("S"), productions=productions)

    weak_normal_form_cfg = cfg_to_weak_normal_form(cfg)

    assert weak_normal_form_cfg.start_symbol == Variable("S")
    assert set(weak_normal_form_cfg.productions) == expected_productions
