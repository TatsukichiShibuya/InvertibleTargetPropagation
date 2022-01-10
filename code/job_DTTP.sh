#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=06:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "grid" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/wruj7p6z
elif [ "${1}" = "bayes" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/jj7sp8a7
fi
