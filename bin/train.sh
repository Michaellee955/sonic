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

MODEL_PATH=${PROJECT_DIR}/model/$1/$2

if [ ! -e ${MODEL_PATH} ]; then
    mkdir -p ${MODEL_PATH}
    echo "made directory " ${MODEL_PATH}
fi


python3 ${PROJECT_DIR}/main.py $1 $MODEL_PATH 3 train

## 把train的load干掉