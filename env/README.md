# Environment reproduction

The eval harness uses **two independent Python environments**, both Python `3.12.3`
and managed with [`uv`](https://docs.astral.sh/uv/). They are intentionally
separate (and conflict on `fsspec`), so each has its own pinned `pyproject.toml`
and committed `uv.lock`:

| Env path (gitignored)   | Project            | Role                                                            |
| ----------------------- | ------------------ | --------------------------------------------------------------- |
| `../.venv`              | `env/golden/`      | Read-only "golden" runtime for perf tasks (torch + tilelang).   |
| `../.venv-agent`        | `env/agent/`       | Agent harness (litellm / openai / mini-swe-agent / datasets).   |

The venvs themselves are multi-GB and gitignored; these lockfiles are the source
of truth for rebuilding them.

## Recreate the environments

Requires `uv` and a CPython 3.12 interpreter on PATH.

```bash
# from the repo root

# golden runtime  -> repo-eval/.venv
( cd env/golden && UV_PROJECT_ENVIRONMENT=../../.venv \
    uv sync --frozen --no-install-project )

# agent harness   -> repo-eval/.venv-agent
( cd env/agent && UV_PROJECT_ENVIRONMENT=../../.venv-agent \
    uv sync --frozen --no-install-project )
```

`--frozen` installs exactly what the `uv.lock` pins; `--no-install-project`
skips the placeholder project (these `pyproject.toml`s only carry dependencies).

The golden env requires a CUDA 13 / NVIDIA GPU host (the locks pin
`linux` + `x86_64` GPU wheels: `torch==2.12.1`, `tilelang==0.1.11`, CUDA-13
`nvidia-*`). The agent env is host-agnostic.

## Updating a lock

If a venv's package set changes, regenerate the matching lock so it stays
authoritative:

```bash
# 1. edit env/<which>/pyproject.toml pins, then:
( cd env/<which> && uv lock )
# 2. re-sync the venv as above
```

## `tilelang` source clone (reference only)

The runtime `tilelang` is the pinned **PyPI wheel `0.1.11`** (in
`env/golden/uv.lock`); the harness imports that installed package.

The repo also keeps a `../tilelang/` source checkout (gitignored) as
open-book reference material for tasks/agents — it is **not** the installed
runtime. To reproduce that checkout:

```bash
git clone https://github.com/tile-ai/tilelang.git
git -C tilelang checkout 03407d5c33747ae0b6a0a3b4530fdca96044a419
```

Note: commit `03407d5` is `v0.1.11-62-g03407d5c`, i.e. 62 commits past the
`v0.1.11` tag — newer than the installed wheel. Keep this distinction in mind
if task code relies on APIs that differ between the two.
