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
if [ "${1}" = "TTT1" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/x5xnj8z8
elif [ "${1}" = "TTT2" ];then
  
fi
