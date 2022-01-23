#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "TTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/9d4r3y9e
fi
