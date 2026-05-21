#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bootstrap_nipoppy_project.sh <STUDY_ROOT> <NIPROOT> [SES]"
  echo "Example: bootstrap_nipoppy_project.sh /path/to/HCP_YA_U100 /path/to/nipoppy-hcpya-u100 01"
  exit 1
fi

STUDY_ROOT="$1"
NIPROOT="$2"
SES="${3:-01}"

RAW="${STUDY_ROOT}/rawdata"
RAW_SES="${STUDY_ROOT}/rawdata_ses"
FS_DERIV="${STUDY_ROOT}/derivatives/freesurfer-7.4.1"   # optional

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] Creating sessioned BIDS view (symlinks): $RAW_SES"
bash "${SCRIPT_DIR}/create_ses_view.sh" --src "$RAW" --dst "$RAW_SES" --ses "$SES"

echo "[2/3] Initializing Nipoppy dataset: $NIPROOT"
mkdir -p "$NIPROOT"
nipoppy init --dataset "$NIPROOT" --bids-source "$RAW_SES"

echo "[3/3] Linking derivatives (if present)"
if [[ -d "$FS_DERIV" ]]; then
  ln -sfn "$FS_DERIV" "$NIPROOT/derivatives/$(basename "$FS_DERIV")"
  echo "  Linked FreeSurfer: $FS_DERIV"
else
  echo "  FreeSurfer not found at: $FS_DERIV (skipping)"
fi

echo "Done."
echo "Nipoppy root: $NIPROOT"
echo "BIDS source (session view): $RAW_SES"
