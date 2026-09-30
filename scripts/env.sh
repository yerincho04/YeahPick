#!/usr/bin/env bash
# Shared configuration for all Act2Answer eval/run scripts. Every script sources this.
# REPO_ROOT is derived from this file's own location (scripts/ lives at the repo root).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# External repos are not vendored. By default they live next to this repo.
export A2A_EXTERNAL_DIR="${A2A_EXTERNAL_DIR:-$(dirname "$REPO_ROOT")}"
export PI0_DEPS_ROOT="${PI0_DEPS_ROOT:-$A2A_EXTERNAL_DIR/RL4VLA}"
export INTERNVLA_REPO="${INTERNVLA_REPO:-$A2A_EXTERNAL_DIR/InternVLA-M1}"
export XIAOMI_REPO="${XIAOMI_REPO:-$A2A_EXTERNAL_DIR/Xiaomi-Robotics-0}"
export MOLMOACT_REPO="${MOLMOACT_REPO:-$A2A_EXTERNAL_DIR/molmoact2}"
export SPATIALVLA_CKPT="${SPATIALVLA_CKPT:-IPEC-COMMUNITY/spatialvla-4b-224-pt}"
export MOLMOACT_CKPT="${MOLMOACT_CKPT:-$A2A_EXTERNAL_DIR/molmoact2_ckpt}"

# Where eval logs are written.
export A2A_LOG_DIR="${A2A_LOG_DIR:-$REPO_ROOT/logs}"; mkdir -p "$A2A_LOG_DIR"
export A2A_OUTPUT_DIR="${A2A_OUTPUT_DIR:-$REPO_ROOT/outputs}"; mkdir -p "$A2A_OUTPUT_DIR"

# This node's network workarounds (authenticating proxy + dead IPv6); harmless elsewhere.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

# ManiSkill prompts "(y|n)" on stdin before downloading a missing task asset (e.g.
# bridge_v2_real2sim for Act2AnswerV4-v1). That prompt has no stdin under sbatch/srun and
# crashes the job with EOFError, so auto-confirm it here for all eval scripts.
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# conda (so `conda activate <env-name>` works inside scripts)
# Where conda envs are created/activated (this node uses a non-default location).
export CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-/home/jovyan/.mlspace/envs}"

if [ -f "${CONDA_ROOT:-/opt/conda}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_ROOT:-/opt/conda}/etc/profile.d/conda.sh"
elif ! type conda >/dev/null 2>&1 && [ -x "${CONDA_ROOT:-}/_conda" ]; then
  # Relocatable/standalone conda installs (as used by this workspace) expose only
  # `_conda`; initialize the shell function inside each eval-script subprocess.
  eval "$("${CONDA_ROOT}/_conda" shell.posix hook)"
fi

# Standard PYTHONPATH for in-process evals (SimplerEnv + ManiSkill; openvla adds $REPO_ROOT/openvla).
export A2A_PYTHONPATH="$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill"
