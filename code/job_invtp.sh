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
if [ "${1}" = "cifar" ];then
  wandb agent tatsukichishibuya/InvTP/vlwrkd07
elif [ "${1}" = "fashion" ];then
  wandb agent tatsukichishibuya/InvTP/x4i1xlib
elif [ "${1}" = "mnist" ];then
  wandb agent tatsukichishibuya/InvTP/9gfvxpzk
fi
