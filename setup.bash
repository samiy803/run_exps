#!/usr/bin/env bash
# setup.bash - Script to set up a Python virtual environment for RL projects
# Usage: ./setup.bash [venv_name] [python_bin]
set -euo pipefail

VENV_NAME="${1:-rl_env}"
PYTHON_BIN="${2:-python3}"

echo ">>> [1/4] Updating APT & installing system packages..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    swig cmake ffmpeg freeglut3-dev xvfb \
    libgl1-mesa-glx libosmesa6-dev patchelf \
    build-essential git python3-venv

echo ">>> [2/4] Creating virtual environment: ${VENV_NAME}"
${PYTHON_BIN} -m venv "${VENV_NAME}"
# shellcheck disable=SC1090
source "${VENV_NAME}/bin/activate"

echo ">>> [3/4] Upgrading pip & installing Python requirements..."
python -m pip install --upgrade pip setuptools wheel

pip install \
  "stable-baselines3[extra,tests,docs]>=2.6.1a1,<3.0" \
  highway-env \
  tensorboard \
  "gymnasium[all]" \
  optuna \
  optunahub \
  wandb \
  optuna-dashboard \
  pyvirtualdisplay

echo ">>> [4/4] Done!  Activate with:"
echo "     source ${VENV_NAME}/bin/activate"
