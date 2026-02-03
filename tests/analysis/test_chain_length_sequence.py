from comext_harmonisation.analysis.chain_length.runner import _chain_length_points


def test_chain_length_sequence_direction():
    points = _chain_length_points(
        min_year=2000, max_year=2003, backward_anchor=2000, forward_anchor=2003
    )
    backward = [p for p in points if p["direction"] == "backward"]
    forward = [p for p in points if p["direction"] == "forward"]

    assert [p["base_year"] for p in backward] == [2001, 2002, 2003]
    assert [p["chain_length"] for p in backward] == [1, 2, 3]

    assert [p["base_year"] for p in forward] == [2000, 2001, 2002]
    assert [p["chain_length"] for p in forward] == [3, 2, 1]
