#!/usr/bin/env bash
# install_pytorch.sh — install the pinned PyTorch CUDA build for the ML_sdfi environment.
# Run after: conda activate ML_sdfi
#
# One stable cu128 build covers Turing (sm_75, e.g. Quadro RTX 8000) through
# Blackwell (sm_120, e.g. RTX PRO 6000 / RTX 50xx), so old and new GPUs share
# the same torch version.
#
# Overrides:
#   PYTORCH_CUDA=cu126              other PyTorch wheel index (same pinned versions)
#   TORCH_VERSION / TORCHVISION_VERSION / TORCHAUDIO_VERSION
#   INSTALL_PYTORCH_NO_GPU=1        skip GPU detection and CUDA smoke test (Docker build)

set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
VARIANT="${PYTORCH_CUDA:-cu128}"

# Lowest compute capability with kernels in the pinned cu128 wheels.
MIN_COMPUTE_CAP="7.5"
# CUDA 12.x minimum driver; below the recommended one, cu128 may still work via minor-version compatibility.
MIN_DRIVER="525.60"
RECOMMENDED_DRIVER="570.0"

version_lt() {
  [[ "$1" != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$1" ]]
}

check_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; CUDA GPU required for ML_sdfi environment" >&2
    exit 1
  fi

  local name cap driver
  while IFS=',' read -r name cap driver; do
    name="$(echo "${name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    cap="$(echo "${cap}" | tr -d ' ')"
    driver="$(echo "${driver}" | tr -d ' ')"
    echo "PYTORCH_INSTALL: gpu=\"${name}\" compute_cap=${cap} driver=${driver}"
    if version_lt "${cap}" "${MIN_COMPUTE_CAP}"; then
      echo "ERROR: ${name} has compute capability ${cap}; torch ${TORCH_VERSION}+${VARIANT} needs >= ${MIN_COMPUTE_CAP}" >&2
      exit 1
    fi
    if version_lt "${driver}" "${MIN_DRIVER}"; then
      echo "ERROR: NVIDIA driver ${driver} is too old for CUDA 12 (need >= ${MIN_DRIVER}); update the driver" >&2
      exit 1
    fi
    if version_lt "${driver}" "${RECOMMENDED_DRIVER}"; then
      echo "WARNING: NVIDIA driver ${driver} is older than ${RECOMMENDED_DRIVER}; ${VARIANT} relies on CUDA minor-version compatibility" >&2
    fi
  done < <(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader)
}

installed_torch() {
  python - <<'PY' 2>/dev/null || echo "none"
import torch
print(torch.__version__)
PY
}

main() {
  local expected current force=()
  expected="${TORCH_VERSION}+${VARIANT}"

  if [[ "${INSTALL_PYTORCH_NO_GPU:-}" != "1" ]]; then
    check_gpus
  fi

  current="$(installed_torch)"
  echo "PYTORCH_INSTALL: selected=${expected} current=${current}"

  if [[ "${current}" != "none" && "${current}" != "${expected}" ]]; then
    echo "Replacing PyTorch (${current} -> ${expected})"
    force=(--force-reinstall)
  fi

  pip install "${force[@]}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${VARIANT}"

  python - <<'PY'
import os
import sys
import torch

print(f"PYTORCH_INSTALL: torch={torch.__version__} cuda={torch.version.cuda}")
print(f"PYTORCH_INSTALL: arch_list={torch.cuda.get_arch_list()}")

if os.environ.get("INSTALL_PYTORCH_NO_GPU") == "1":
    print("PYTORCH_INSTALL: skipping CUDA smoke test (INSTALL_PYTORCH_NO_GPU=1)")
    sys.exit(0)

if not torch.cuda.is_available():
    print("CUDA_UNAVAILABLE: torch.cuda.is_available() is False after install", file=sys.stderr)
    sys.exit(1)

arch_list = set(torch.cuda.get_arch_list())
for idx in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    sm = f"sm_{major}{minor}"
    # A wheel without kernels for this GPU still reports is_available() == True.
    if sm not in arch_list and f"compute_{major}{minor}" not in arch_list:
        print(f"CUDA_UNAVAILABLE: {name} ({sm}) not in torch arch list {sorted(arch_list)}", file=sys.stderr)
        sys.exit(1)
    try:
        value = (torch.ones(1024, device=f"cuda:{idx}") * 2).sum().item()
    except Exception as exc:
        print(f"CUDA_UNAVAILABLE: kernel launch failed on {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    if value != 2048.0:
        print(f"CUDA_UNAVAILABLE: wrong result on {name}: {value}", file=sys.stderr)
        sys.exit(1)
    print(f"PYTORCH_INSTALL: device={idx} {name} ({sm}) kernel_ok")
PY
}

main "$@"
