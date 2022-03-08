#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:20:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "cifar" ];then
  wandb agent tatsukichishibuya/InvTP/f1zvys2d
elif [ "${1}" = "fashion" ];then
  wandb agent tatsukichishibuya/InvTP/gug0eiqc
elif [ "${1}" = "mnist" ];then
  wandb agent tatsukichishibuya/InvTP/5x3lmzva
elif [ "${1}" = "cifar100" ];then
  wandb agent tatsukichishibuya/InvTP/aq2sivfl
fi
