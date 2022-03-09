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
if [ "${1}" = "mnist" ];then
  wandb agent tatsukichishibuya/InvTP/10z31ly1
elif [ "${1}" = "fashion" ];then
  wandb agent tatsukichishibuya/InvTP/5clamq3h
elif [ "${1}" = "cifar10" ];then
  wandb agent tatsukichishibuya/InvTP/hw6pfrun
elif [ "${1}" = "cifar100" ];then
  wandb agent tatsukichishibuya/InvTP/ner0aqmn
fi
