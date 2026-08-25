#!/usr/bin/env bash
set -euo pipefail

environment_name="${1:-${CONDA_DEFAULT_ENV:-python}}"
output_path="${2:-environment_reports/${environment_name}.txt}"
mkdir -p "$(dirname "${output_path}")"

{
    echo "environment=${environment_name}"
    echo "collected_at=$(date --iso-8601=seconds)"
    echo

    echo "[operating_system]"
    if [[ -f /etc/os-release ]]; then
        cat /etc/os-release
    fi
    echo "architecture=$(uname -m)"
    echo

    echo "[cpu_and_memory]"
    lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket):' || true
    free -h || true
    echo

    echo "[gpu]"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version \
            --format=csv,noheader
    else
        echo "nvidia-smi not available"
    fi
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    fi
    echo

    echo "[python]"
    python --version
    python -m pip --version
    python - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch=not installed")
PY
    echo

    echo "[installed_packages]"
    python -m pip freeze
} > "${output_path}"

echo "Environment report written to ${output_path}"
