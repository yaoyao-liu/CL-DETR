#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
JOB_TIME=$4
RUN_COMMAND=${@:5}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

sbatch -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -t ${JOB_TIME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    ${SRUN_ARGS} \
    ${RUN_COMMAND}

