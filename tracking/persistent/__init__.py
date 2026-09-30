"""Persistent actor reconstruction built on top of canonical Tracking V1.

The persistent layer never rewrites Tracking V1 physical identities.  It uses
those identities as conservative evidence fragments and builds a second layer
whose goal is long-lived actor continuity.
"""

from .candidates import (
    CANDIDATE_VERSION,
    PersistentActorCandidateConfig,
    build_candidate_diagnostics,
    estimate_population,
)

__all__ = [
    "CANDIDATE_VERSION",
    "PersistentActorCandidateConfig",
    "build_candidate_diagnostics",
    "estimate_population",
]
