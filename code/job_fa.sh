#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:50:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "cifar" ];then
  wandb agent tatsukichishibuya/InvTP/9985drd7
elif [ "${1}" = "fashion" ];then
  wandb agent tatsukichishibuya/InvTP/70vxkqo7
elif [ "${1}" = "mnist" ];then
  wandb agent tatsukichishibuya/InvTP/o8yoql2a
fi
