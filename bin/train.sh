#!/usr/bin/bash

cd $(dirname $0)/..
PROJECT_DIR=$(pwd)

# if [ $# -lt 2 ]; then
#     echo "<Usage>: run.sh model_type model_name [env_idx]"
#     echo "model_type: lstm or cnn"
#     echo "model_path: relative path from root directory of project"
#     echo "env_idx: index of environment to play with (-1 to 12)"
#     echo "          default: -1 (random environment)"
#     exit 1
# fi

# MODEL_PATH=${PROJECT_DIR}/model/$1/checkpoints/$2

# if [ ! -f ${MODEL_PATH} ]; then
#     echo "ERROR: " ${MODEL_PATH} " does not exist"
#     exit 1
# fi


python3 ${PROJECT_DIR}/main.py | tee ${PROJECT_DIR}/log/$1/train_log.txt
