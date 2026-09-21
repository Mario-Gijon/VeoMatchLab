# Analytics service

This directory marks the future boundary for GPU-backed match analysis. It is
not implemented yet. A future service will receive match identifiers and use
the shared local match-data root, rather than receiving full videos over HTTP.

It may import `geometry`, but must remain separate from `web_app`.
