#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u visualize_erf.py ${PY_ARGS}
