"""Identity-stage building blocks for Tracking V1.

The package is intentionally split into:
- candidate/compatibility construction (Hito 3A),
- global identity optimization (future Hito 3B+).

Hito 3A never assigns a physical identity.
"""

from .candidates import (
    CandidateEdge,
    CandidateGraph,
    PhysicalCompatibilityConfig,
    TrackletNode,
    TrackletObservation,
    build_candidate_edge,
    build_candidate_graph,
    build_candidate_graph_summary,
    iter_tracklet_nodes_jsonl,
    load_appearance_pairwise_jsonl,
)

__all__ = [
    "CandidateEdge",
    "CandidateGraph",
    "PhysicalCompatibilityConfig",
    "TrackletNode",
    "TrackletObservation",
    "build_candidate_edge",
    "build_candidate_graph",
    "build_candidate_graph_summary",
    "iter_tracklet_nodes_jsonl",
    "load_appearance_pairwise_jsonl",
]
