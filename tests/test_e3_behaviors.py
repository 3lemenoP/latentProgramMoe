"""Ground-truth E3 / E3′ seq2seq behaviors (order and append)."""
from lpm.tasks import E3Vocab, e3_output

V = E3Vocab()
X = [V.N_SPECIAL + i for i in range(5)]


def test_order_differs():
    assert e3_output(X, "a_then_b", V) == list(reversed(X)) + [V.A]
    assert e3_output(X, "b_then_a", V) == [V.A] + list(reversed(X))
    assert e3_output(X, "a_then_b", V) != e3_output(X, "b_then_a", V)


def test_append_composes_with_reverse():
    assert e3_output(X, "append", V) == X + [V.B]
    assert e3_output(X, "c_then_b", V) == [V.B] + list(reversed(X))
    assert e3_output(X, "b_then_c", V) == list(reversed(X)) + [V.B]
    assert e3_output(X, "c_then_b", V) != e3_output(X, "b_then_c", V)
