#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=23:55:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "lr_ratio" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/t49y28lk
else
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/m48rv2it
fi
