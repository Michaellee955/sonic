#!/usr/bin/bash

cd $(dirname $0)/..
PROJECT_DIR=$(pwd)

if [ $# -lt 2 ]; then
    echo "<Usage>: run.sh model_type model_name [env_idx]"
    echo "model_type: lstm or cnn"
    echo "model_path: relative path from root directory of project"
    echo "env_idx: index of environment to play with (-1 to 12)"
    echo "          default: -1 (random environment)"
    exit 1
fi

MODEL_PATH=${PROJECT_DIR}/model/$1/checkpoints/$2

if [ ! -f ${MODEL_PATH} ]; then
    echo "ERROR: " ${MODEL_PATH} " does not exist"
    exit 1
fi

if [ $# -eq 3 ]; then
    python3 ${PROJECT_DIR}/runner.py $1 ${MODEL_PATH} $3
else
    python3 ${PROJECT_DIR}/runner.py $1 ${MODEL_PATH} -1
fi