#!/usr/bin/env bash
set -uxo pipefail

# export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
# export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
# export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
# export OVERWRITE=${OVERWRITE:-0}

# mkdir -p "${VERL_HOME}/data"
TRAIN_FILE=/home/hhzhang/improve/verl/datasets/dapo/dapo-math-17k.parquet
TEST_FILE=/home/hhzhang/improve/verl/datasets/aime/aime-2024.parquet
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"


  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
