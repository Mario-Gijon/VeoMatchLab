#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT}/.venv-prtreid"
MODEL_DIR="${ROOT}/models/prtreid"

CHECKPOINT="${MODEL_DIR}/prtreid-soccernet-baseline.pth.tar"
HRNET="${MODEL_DIR}/hrnetv2_w32_imagenet_pretrained.pth"

CHECKPOINT_URL="https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
HRNET_URL="https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1"

CHECKPOINT_MD5="9633825232bc89f23a94522c5561650e"
HRNET_MD5="58ea12b0420aa3adaa2f74114c9f9721"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required for the isolated PRTReID environment."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required to download PRTReID weights."
  exit 1
fi

if ! command -v md5sum >/dev/null 2>&1; then
  echo "ERROR: md5sum is required to verify PRTReID weights."
  exit 1
fi

mkdir -p "${MODEL_DIR}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[setup-prtreid] creating Python 3.10 environment at ${ENV_DIR}"
  uv venv --python 3.10 "${ENV_DIR}"
else
  echo "[setup-prtreid] reusing existing environment at ${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"

echo "[setup-prtreid] python=$("${PYTHON}" --version 2>&1)"

# Check installation without importing prtreid itself, because an incompatible
# third-party import is exactly what this setup script repairs below.
PRTREID_PRESENT="$("${PYTHON}" - <<'PY'
import importlib.util
print("yes" if importlib.util.find_spec("prtreid") is not None else "no")
PY
)"

if [[ "${PRTREID_PRESENT}" != "yes" ]]; then
  echo "[setup-prtreid] installing PRTReID"
  uv pip install \
    --python "${PYTHON}" \
    "prtreid @ git+https://github.com/VlSomers/prtreid"
else
  echo "[setup-prtreid] PRTReID package already present"
fi

echo "[setup-prtreid] aligning runtime dependencies"
uv pip install \
  --python "${PYTHON}" \
  "numpy==1.26.4" \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "scikit-image==0.24.0"

# PRTReID commit 30617a75967e84d5d516959c4b84cbeea6f56493 imports
# `functional` from albumentations in random_occlusion.py, but that symbol is
# never used by the file. Albumentations 2.x no longer exposes it at the
# package root. Remove only that stale, unused import.
echo "[setup-prtreid] applying Albumentations 2 compatibility patch"
"${PYTHON}" - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("prtreid")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit("ERROR: could not locate installed prtreid package")

package_root = Path(next(iter(spec.submodule_search_locations)))
target = (
    package_root
    / "data"
    / "data_augmentation"
    / "random_occlusion.py"
)

if not target.is_file():
    raise SystemExit(f"ERROR: PRTReID compatibility target missing: {target}")

text = target.read_text(encoding="utf-8")

old_multiline = """from albumentations import (
    DualTransform, functional
)
"""
new_multiline = """from albumentations import DualTransform
"""

if old_multiline in text:
    text = text.replace(old_multiline, new_multiline, 1)
    target.write_text(text, encoding="utf-8")
    print(f"[setup-prtreid] patched {target}")
elif "DualTransform, functional" in text:
    raise SystemExit(
        "ERROR: PRTReID random_occlusion import changed shape; "
        "refusing an unsafe automatic patch"
    )
elif "from albumentations import DualTransform" in text:
    print(f"[setup-prtreid] compatibility patch already applied: {target}")
else:
    raise SystemExit(
        "ERROR: unexpected PRTReID random_occlusion import; "
        "inspect before patching"
    )
PY

download_and_verify() {
  local url="$1"
  local output="$2"
  local expected_md5="$3"

  if [[ ! -f "${output}" ]]; then
    echo "[setup-prtreid] downloading $(basename "${output}")"
    curl -L --fail --retry 3 "${url}" -o "${output}"
  else
    echo "[setup-prtreid] using existing $(basename "${output}")"
  fi

  local actual_md5
  actual_md5="$(md5sum "${output}" | awk '{print $1}')"

  if [[ "${actual_md5}" != "${expected_md5}" ]]; then
    echo "ERROR: md5 mismatch for ${output}"
    echo "expected=${expected_md5}"
    echo "actual=${actual_md5}"
    exit 1
  fi
}

download_and_verify \
  "${CHECKPOINT_URL}" \
  "${CHECKPOINT}" \
  "${CHECKPOINT_MD5}"

download_and_verify \
  "${HRNET_URL}" \
  "${HRNET}" \
  "${HRNET_MD5}"

echo "[setup-prtreid] smoke-import"
"${PYTHON}" - <<'PY'
import albumentations
import numpy
import skimage
import torch
import torchvision
import prtreid

print("albumentations:", albumentations.__version__)
print("numpy:", numpy.__version__)
print("skimage:", skimage.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("compute_capability:", torch.cuda.get_device_capability(0))

    x = torch.ones((32, 32), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_smoke_sum:", float(y.sum().cpu()))

print("prtreid:", getattr(prtreid, "__version__", "unknown"))
PY

echo
echo "[setup-prtreid] ready"
echo "python=${PYTHON}"
echo "checkpoint=${CHECKPOINT}"
echo "hrnet=${HRNET}"
