#!/usr/bin/env bash
# Register all GPE serial-section samples, SEQUENTIALLY.
# brainreg is internally multi-process + loads the atlas into RAM, so running several
# at once OOMs the box. One at a time, each using the machine (capped via --n-free-cpus).
#
# Env knobs:
#   SOURCE=full|downsampled   (default full = stitched ch + recipe voxel; the real pass)
#   RES=25                     (µm, only when SOURCE=downsampled)
#   NFREE=8                    (brainreg --n-free-cpus; 32-8 = 24 threads)
set -u

export DP=/mnt/fastdata/code/datapipe/.venv/bin/datapipe
export BR="$HOME/.venvs/brainglobe/bin/brainreg"
export ANALYSIS=/ceph/scratch/larsr/analysis_gpe
export LOGS=/mnt/fastdata/code/datapipe/.runlogs/gpe
export SRC=/ceph/scratch/larsr/tmpgpe
export SOURCE="${SOURCE:-full}"
export RES="${RES:-25}"
export NFREE="${NFREE:-8}"
mkdir -p "$LOGS"

run_one() {
  local d="$1" s; s="$(basename "$d")"
  local src=(--source "$SOURCE")
  [ "$SOURCE" = downsampled ] && src+=(--res "$RES")
  echo "[start] $s ($SOURCE)"
  if "$DP" register "$d" "${src[@]}" --orientation psr \
        --analysis-root "$ANALYSIS" --brainreg "$BR" --overwrite \
        --brainreg-extra "--n-free-cpus $NFREE" \
        > "$LOGS/$s.log" 2>&1; then
    echo "[done]  $s"
  else
    echo "[FAIL]  $s — see $LOGS/$s.log"
  fi
}
export -f run_one

ls -d "$SRC"/*/ | xargs -P 1 -I{} bash -c 'run_one "$@"' _ {}
echo "ALL DONE — logs in $LOGS"
