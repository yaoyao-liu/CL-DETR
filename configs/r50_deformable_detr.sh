#!/usr/bin/env bash

__conda_setup="$('/home/yaliu/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yaliu/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yaliu/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/yaliu/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source activate deformable_detr

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
