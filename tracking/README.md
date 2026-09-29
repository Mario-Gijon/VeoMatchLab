# Tracking

This directory intentionally starts empty after the V0 freeze.  The historical
implementation lives in `tracking_v0/` and is retained only for compatibility
with existing Baseline, V2.8, and V8 workflows.

New tracking must begin from a clean architecture:

`detections -> short-term tracklets -> identity evidence -> global offline identity reconstruction`

Do not automatically reuse V0 code from the new tracking namespace.
