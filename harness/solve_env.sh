#!/usr/bin/env bash
# Hardening sourced into the AGENT's shell during the solve phase (in addition
# to the workspace env.sh). Goal: stop the agent from fetching the upstream fix
# or otherwise reaching the network. NOTE: without root/userns we cannot create
# a real network namespace on this host, so this is best-effort (blocks pip,
# git, curl, and well-behaved HTTP clients; a determined raw-socket bypass is
# still possible -- documented as a limitation).
#
# Usage (in the agent runner):  source <work>/env.sh; source harness/solve_env.sh

# 1) Scrub credentials / API tokens so any network tool is unauthenticated.
for v in GH_TOKEN GITHUB_TOKEN GITHUB_API_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN \
         HUGGINGFACE_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY WANDB_API_KEY \
         AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
         GOOGLE_API_KEY GCP_API_KEY NETRC; do
  unset "$v" 2>/dev/null || true
done

# 2) Point all proxies at a dead endpoint so HTTP(S) clients fail fast.
export HTTP_PROXY="http://127.0.0.1:9" HTTPS_PROXY="http://127.0.0.1:9" ALL_PROXY="http://127.0.0.1:9"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
export NO_PROXY="" no_proxy=""

# 3) Force package managers / git offline.
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_OFFLINE=1 UV_NO_INDEX=1
export GIT_TERMINAL_PROMPT=0
export GIT_ALLOW_PROTOCOL="file"   # block http/https/ssh/git transports
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 4) Empty any credential helper / netrc lookups.
export GIT_CONFIG_NOSYSTEM=1
export HOME="${HOME:?solve_env must be sourced AFTER the workspace env.sh}"

echo "[solve_env] credentials scrubbed; network forced offline (best-effort, no netns on this host)" >&2
