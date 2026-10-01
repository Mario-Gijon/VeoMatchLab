from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


@dataclass
class IdentityNode:
    identity_id: str
    family_key: str | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    embedding: List[float] | None = None


@dataclass
class IdentityEdge:
    left: str
    right: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


class IdentityGraph:
    def __init__(self):
        self.nodes: Dict[str, IdentityNode] = {}
        self.edges: List[IdentityEdge] = []

    def add_node(self, node: IdentityNode):
        self.nodes[node.identity_id] = node

    def add_edge(self, edge: IdentityEdge):
        self.edges.append(edge)

    def connected_components(
        self,
        min_score: float = 0.0,
    ) -> List[List[str]]:
        adjacency = defaultdict(list)

        for edge in self.edges:
            if edge.score >= min_score:
                adjacency[edge.left].append(edge.right)
                adjacency[edge.right].append(edge.left)

        visited = set()
        components = []

        for node_id in self.nodes:
            if node_id in visited:
                continue

            stack = [node_id]
            component = []

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbour in adjacency[current]:
                    if neighbour not in visited:
                        stack.append(neighbour)

            components.append(sorted(component))

        return components


def build_identity_score(
    appearance: float,
    family_bonus: float,
    temporal_score: float,
    motion_score: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Score global de unión entre dos identidades.
    """

    score = (
        0.55 * appearance
        + 0.20 * family_bonus
        + 0.15 * temporal_score
        + 0.10 * motion_score
    )

    return score, {
        "appearance": appearance,
        "family": family_bonus,
        "temporal": temporal_score,
        "motion": motion_score,
    }


def can_link(
    left: IdentityNode,
    right: IdentityNode,
) -> bool:
    """
    Restricciones duras iniciales.
    """

    if (
        left.frame_start is not None
        and right.frame_start is not None
        and left.frame_end is not None
        and right.frame_end is not None
    ):
        overlap = (
            max(
                0,
                min(left.frame_end, right.frame_end)
                - max(left.frame_start, right.frame_start)
            )
        )

        if overlap > 0:
            return False

    return True
