#!/usr/bin/env bash
# Install the shared-environment repos in editable mode.
#
# Identical in all four repos, so it cannot assume which one it is run from:
# it installs every repo of the group by name, including the current one, and
# skips any that are not checked out.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SHARED_REPOS=(
  ML_geo_production
  multi_channel_dataset_creation
  ML_sdfi_fastai2
  ML_Production
)

install_sibling() {
  local name="$1"
  local dir="${PARENT}/${name}"
  if [[ -d "${dir}" ]]; then
    echo "INSTALL_LOCAL: ${name}"
    (cd "${dir}" && pip install -e .)
  else
    echo "INSTALL_LOCAL: skip ${name} (not found at ${dir})"
  fi
}

for repo in "${SHARED_REPOS[@]}"; do
  install_sibling "${repo}"
done
