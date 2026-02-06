#!/usr/bin/env bash
set -euo pipefail
export KMP_USE_SHM=0
export KMP_DISABLE_SHM=1
export OMP_NUM_THREADS=1

ROOT_DIR="/home/ubuntu/donghoon/projects/gaze"
SCRIPT="${ROOT_DIR}/DeepGaze/scanpath_visualize.py"
IMAGE_DIR="${ROOT_DIR}/ReNO/sample_images"
OUT_ROOT="${ROOT_DIR}/DeepGaze/scanpath_outputs_iie"

for points in 3 4 5; do
  python "${SCRIPT}" \
    --image-dir "${IMAGE_DIR}" \
    --out-root "${OUT_ROOT}/points_${points}" \
    --model iie \
    --num-points "${points}"
done
