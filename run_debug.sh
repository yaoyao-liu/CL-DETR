#!/bin/sh

CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr.sh