# Agent Guidelines -- study before each reasoning process

- Keep the project simple, fast, efficient, reproducible.
- Work with high quality code and maintain an excellent development hygiene.
- Implement an object-oriented design
- No unrelated files (README.md, SETUP.md,etc.).
- Never create demo data or demo files
- Always check for unused imports
- Avoid redundant code
- Prioritize pyproject.toml files opposed to requirements.txt files
- Secrets live in C:/.../WORKDIR/.../.env
- API response type: ...
- Cache root: C:/.../WORKDIR/.../cache/
- All inline comments and docstrings in English.

## Purpose

A concise, enforceable development outline for this repository. It defines how we design, code, test, document, and operate the system so work remains simple, fast, efficient, and reproducible.

## Core Principles

- **KISS & focus**: Prefer the simplest workable solution. Avoid speculative features and abstractions.
- **Quality first**: Clean code, small PRs, frequent commits, and rigorous checks.
- **Reproducibility**: One command should set up the environment and run the project.
- **Single source of truth**: Configuration and contracts are centralized and typed.

## Repository Hygiene

- No unrelated boilerplate (marketing docs, demo data/files, screenshots). Keep `README.md` and `docs/` strictly project‑relevant.
- Enforce a lean tree:

  "
  project/
    src/                # production code
    tests/              # test suite
    docs/               # documentation
    cache/              # runtime and longterm cache (gitignored)
    .env.example        # template of required env vars (no secrets)
    pyproject.toml      # build + tooling config
  "
  
- Do **not** commit large binaries, credentials, or vendor dumps. Use `.gitignore` accordingly.

## Architecture & Design

- Object‑oriented design with **SOLID** in mind; favor composition over inheritance.
- Small, cohesive modules; no god classes. Public APIs are explicit and minimal.
- External boundaries (I/O, APIs, clock, randomness) are wrapped behind interfaces for testability.

## Configuration & Secrets

- Secrets live in the project **workdir** `.env` file and are never committed.
  - POSIX: `${WORKDIR}/.env`  
    Windows: `C:\…\WORKDIR\…\.env`
- Provide `.env.example` with variable names and safe placeholders.
- Load configuration via a typed settings object (e.g., `pydantic`/`dataclass`) with clear defaults and validation.
- **Environment variables win** over file defaults. No hardcoded secrets in code or tests.

### Time format polify

- We lock a single canonical time representation across the whole codebase.
- Serialization: ISO8601 UTC strings: <YYYY-MM-DDTHH:MM:SSZ> respectively <YYYY-MM-DDTHH:MM:SS±HH:MM>

## API Contracts

- Define explicit response models (e.g., `ApiResponse[T]`) instead of returning untyped dicts.
- Each model documents fields, units, and constraints (e.g., prices > 0, quantities finite, timezone = UTC).
- All network calls have timeouts, retries with backoff, and structured error handling.

## Caching

- Cache root is `${WORKDIR}/cache/` (Windows: `C:\…\WORKDIR\…\cache\`). Ensure the directory is created at startup.
- Name cache keys deterministically. Define TTL/invalidation policy per cache.
- Cache is an optimization, not a source of truth; corrupted entries must fail safely and be rebuilt.

## Dependencies & Tooling

- Always remove unused imports; enforced by linter.

## Coding Standards

- All inline comments and docstrings are **English** and explain *why*, not just *what*.
- Type hints are mandatory at public boundaries; mypy must pass with `--strict` (or project policy).
- Prefer pure functions for business logic; isolate side effects.
- Time handling is **UTC** internally; convert at the edges.
- Prefer polars over pandas. Polars scales better and performs faster on large datasets.

## Logging & Observability

- Structured logging (JSON where possible). Default `LOG_LEVEL=INFO`; override via env.
- Never log secrets or full payloads containing sensitive data.
- Add metrics/hooks where it improves operability (timers, counters, error rates).

## Git Workflow

- Commit only meaningful, buildable states.

### Commit Message Guidelines

We use the following format for commit headers:

#### Types

- **feat**: new feature
- **fix**: bug fix
- **chore**: repo maintenance, config, tooling
- **docs**: documentation only
- **refactor**: code refactoring (no new features, no fixes)
- **test**: adding or modifying tests

#### Scopes

- **backend**, **backtest**, **bot**, **common**, **data**, **frontend**, **tests**, **streamlit**, **docker**, **kubernetes**, **miscellaneous**

#### Examples

- `[feat|backend] Add order execution endpoint`  
- `[chore|docker] Add .dockerignore file`
- `[refactor|common] Simplify logging config`

## Documentation

- `README.md` covers: purpose, quickstart, commands, env vars, and troubleshooting.
- Each module has a short overview in its docstring. Longer guides go to `docs/`.

## Domain Notes (Algorithmic Trading)

- Separate **backtesting** and **live trading** code paths behind clear interfaces.
- Single clock source; trading calendar awareness (holidays/early closes).
- Reproducible backtests: pinned data snapshots, fixed seeds, frozen config.
- Risk management is non‑negotiable: position limits, max drawdown guards, and input validation.

---
**Non‑negotiables checklist (quick scan):**

- Simple, deterministic, reproducible.
- Typed configs and API models; no hardcoded secrets.
- No demo data/files in the repo.
- Unused imports are removed; linters clean.
- English comments/docstrings.
