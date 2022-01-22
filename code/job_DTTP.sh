#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "TTT5" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/13uqr8t7
elif [ "${1}" = "TTT7" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/r9fyv32v
fi
