"""Offline actor-reconstruction primitives for VeoMatchLab.

This package sits above conservative local tracklets. It must not turn a
tracklet into an identity by itself: it only exposes sparse, explainable
evidence for later global/offline solving.
"""

from .index import (
    ACTOR_INDEX_VERSION,
    CoobservedPairEvidence,
    EndpointVelocity,
    HardConflict,
    SparseTrackletIndex,
    SparseTrackletIndexConfig,
    SparseTrackletNode,
    TransitionFeatures,
    build_sparse_tracklet_index,
)

__all__ = [
    "ACTOR_INDEX_VERSION",
    "CoobservedPairEvidence",
    "EndpointVelocity",
    "HardConflict",
    "SparseTrackletIndex",
    "SparseTrackletIndexConfig",
    "SparseTrackletNode",
    "TransitionFeatures",
    "build_sparse_tracklet_index",
]
