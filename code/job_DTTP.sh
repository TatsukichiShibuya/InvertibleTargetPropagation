#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "CCC1" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/bz1oikpi
elif [ "${1}" = "CCC2" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/fjw4745o
fi
