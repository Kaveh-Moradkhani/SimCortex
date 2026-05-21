set -euo pipefail
shopt -s nullglob

usage() {
  cat << EOF
Usage:
  create_ses_view.sh --src <BIDS_ROOT_NO_SES> --dst <BIDS_ROOT_WITH_SES> [--ses 01]

What it does:
  Creates a session-aware BIDS "view" using symlinks:
    sub-XXXX/<datatype>/sub-XXXX_<suffix>.*  -->
    sub-XXXX/ses-01/<datatype>/sub-XXXX_ses-01_<suffix>.*

Notes:
  - Non-destructive: original dataset is not modified.
  - Only files are linked (no copying).
EOF
}

SRC=""
DST=""
SES="01"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --dst) DST="$2"; shift 2 ;;
    --ses) SES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SRC" || -z "$DST" ]]; then
  echo "ERROR: --src and --dst are required."
  usage
  exit 1
fi

mkdir -p "$DST"

for f in "$SRC"/*; do
  bn="$(basename "$f")"
  [[ "$bn" == sub-* ]] && continue
  ln -sfn "$f" "$DST/$bn"
done

KNOWN_DT=("anat" "func" "dwi" "fmap" "perf" "pet" "beh" "eeg" "meg" "ieeg" "nirs" "micr" "motion" "mrs")

is_known_dt() {
  local x="$1"
  for dt in "${KNOWN_DT[@]}"; do
    [[ "$dt" == "$x" ]] && return 0
  done
  return 1
}

for sub in "$SRC"/sub-*; do
  sid="$(basename "$sub")"
  mkdir -p "$DST/$sid/ses-$SES"

  found_dt=0
  for d in "$sub"/*; do
    [[ "$(basename "$d")" == ses-* ]] && continue
    [[ -d "$d" ]] || continue
    dt="$(basename "$d")"
    if is_known_dt "$dt"; then
      found_dt=1
      mkdir -p "$DST/$sid/ses-$SES/$dt"
      for f in "$d"/*; do
        [[ -f "$f" ]] || continue
        bn="$(basename "$f")"

        if [[ "$bn" == ${sid}_* && "$bn" != *"_ses-"* ]]; then
          new="${bn/#${sid}_/${sid}_ses-${SES}_}"
        else
          new="$bn"
        fi
        ln -sfn "$f" "$DST/$sid/ses-$SES/$dt/$new"
      done
    fi
  done

  if [[ "$found_dt" -eq 0 ]]; then
    mkdir -p "$DST/$sid/ses-$SES/anat"
    for f in "$sub"/*; do
      [[ -f "$f" ]] || continue
      bn="$(basename "$f")"
      if [[ "$bn" == ${sid}_* && "$bn" != *"_ses-"* ]]; then
        new="${bn/#${sid}_/${sid}_ses-${SES}_}"
      else
        new="$bn"
      fi
      ln -sfn "$f" "$DST/$sid/ses-$SES/anat/$new"
    done
  fi
done

echo "Done. Sessioned BIDS view at: $DST"
