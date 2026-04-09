# Security Policy

## Reporting a Vulnerability

Please open a private GitHub security advisory or contact the maintainer directly before public disclosure. Include the affected deployment path, reproduction steps, and environment details such as Redis or Kubernetes configuration when relevant.

## Audit Summary

- No hardcoded tokens, secrets, or cloud credentials were found during the April 2026 audit.
- No user-controlled shell execution, unsafe YAML loading, or SQL injection patterns were identified.
- The FastAPI surface validates request payloads and the queue/router logic keeps all work in-process; no untrusted pickle deserialization paths were found.
- Dependency and static analysis should be re-run with `pip-audit -r requirements.txt` and `bandit -r . -ll` whenever serving or deployment dependencies change.
