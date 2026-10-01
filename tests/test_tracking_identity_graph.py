from tracking.actors.identity_graph import (
    IdentityGraph,
    IdentityNode,
    IdentityEdge,
    build_identity_score,
    can_link,
)


def test_score_combination():
    score, parts = build_identity_score(
        appearance=1.0,
        family_bonus=1.0,
        temporal_score=1.0,
        motion_score=1.0,
    )

    assert score == 1.0
    assert parts["appearance"] == 1.0


def test_components_merge():
    graph = IdentityGraph()

    graph.add_node(IdentityNode("A"))
    graph.add_node(IdentityNode("B"))
    graph.add_node(IdentityNode("C"))

    graph.add_edge(
        IdentityEdge(
            "A",
            "B",
            0.9,
        )
    )

    components = graph.connected_components()

    assert ["A", "B"] in components
    assert ["C"] in components


def test_low_score_not_connected():
    graph = IdentityGraph()

    graph.add_node(IdentityNode("A"))
    graph.add_node(IdentityNode("B"))

    graph.add_edge(
        IdentityEdge(
            "A",
            "B",
            0.2,
        )
    )

    components = graph.connected_components(
        min_score=0.5
    )

    assert ["A"] in components
    assert ["B"] in components


def test_overlap_blocks_link():
    a = IdentityNode(
        "A",
        frame_start=0,
        frame_end=100,
    )

    b = IdentityNode(
        "B",
        frame_start=50,
        frame_end=150,
    )

    assert not can_link(a, b)


def test_non_overlap_allows_link():
    a = IdentityNode(
        "A",
        frame_start=0,
        frame_end=100,
    )

    b = IdentityNode(
        "B",
        frame_start=120,
        frame_end=150,
    )

    assert can_link(a, b)
