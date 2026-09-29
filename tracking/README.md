# Tracking

This directory starts the clean Tracking V1 architecture. The historical
implementation remains isolated in `tracking_v0/`; do not automatically reuse
it from this namespace.

`CanonicalDetection` is one physical person at one instant, with no temporal
identity. The pipeline is:

`detections -> tracklets -> appearance/evidence -> identities -> final`

Fragmentation is preferable to contaminating a tracklet with different people.
V1 explicitly expects missed detections, players entering/leaving the FOV,
occlusions, and dense corners or free kicks. Local MOT must not force an
identity across long disappearances: later global reconstruction joins
tracklets only with sufficient evidence. Veo X/Z geometry is fundamental
evidence for that reconstruction.
