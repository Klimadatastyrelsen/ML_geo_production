#!/usr/bin/env bash
# install_pytorch.sh — auto-select PyTorch CUDA build for this machine.
# Run after: conda activate ML_sdfi
# Override: PYTORCH_CUDA=cu124|cu121|cu128-nightly

set -euo pipefail

detect_variant() {
  if [[ -n "${PYTORCH_CUDA:-}" ]]; then
    echo "${PYTORCH_CUDA}"
    return
  fi
  if [[ "${INSTALL_PYTORCH_NO_GPU:-}" == "1" ]]; then
    echo "cu124"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; CUDA GPU required for ML_sdfi environment" >&2
    exit 1
  fi
  local cap name
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  if [[ "${cap}" == "12.0" ]] || echo "${name}" | grep -qiE 'Blackwell|RTX 50'; then
    echo "cu128-nightly"
  else
    echo "cu124"
  fi
}

current_torch_variant() {
  python - <<'PY' 2>/dev/null || echo "none"
import torch
v = getattr(torch.version, "cuda", None) or "none"
if v in ("none", None):
    print("none")
elif "12.8" in str(v) or str(v).startswith("13."):
    print("cu128-nightly")
else:
    print("cu124")
PY
}

main() {
  local variant current force=()
  variant="$(detect_variant)"
  current="$(current_torch_variant)"

  echo "PYTORCH_INSTALL: selected=${variant} current=${current}"

  if [[ "${current}" != "${variant}" && "${current}" != "none" ]]; then
    echo "Replacing PyTorch (${current} -> ${variant})"
    force=(--force-reinstall)
  fi

  case "${variant}" in
    cu128-nightly)
      pip install "${force[@]}" --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128
      ;;
    cu124)
      pip install "${force[@]}" torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
      ;;
    cu121)
      pip install "${force[@]}" torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
      ;;
    *)
      echo "ERROR: unknown PYTORCH_CUDA=${variant} (use cu124, cu121, or cu128-nightly)" >&2
      exit 1
      ;;
  esac

  python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available()
print(f"PYTORCH_INSTALL: cuda_available={ok}")
print(f"PYTORCH_INSTALL: torch={torch.__version__} cuda={torch.version.cuda}")
if ok:
    print(f"PYTORCH_INSTALL: device={torch.cuda.get_device_name(0)}")
elif __import__("os").environ.get("INSTALL_PYTORCH_NO_GPU") == "1":
    print("PYTORCH_INSTALL: skipping CUDA smoke test (INSTALL_PYTORCH_NO_GPU=1)")
else:
    print("CUDA_UNAVAILABLE: torch.cuda.is_available() is False after install", file=sys.stderr)
    sys.exit(1)
PY
}

main "$@"
