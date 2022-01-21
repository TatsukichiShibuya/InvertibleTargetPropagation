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
if [ "${1}" = "CCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/umulf3vp
elif [ "${1}" = "TTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/m614upgx
elif [ "${1}" = "CCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/xhvs6h87
elif [ "${1}" = "TTT1" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/e6gxk87s
elif [ "${1}" = "TTT2" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/ypafcd7o
fi
