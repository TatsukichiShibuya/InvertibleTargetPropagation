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
if [ "${1}" = "CCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/laf3350d
elif [ "${1}" = "CCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/08kjgdq0
elif [ "${1}" = "CTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/n5tmufzl
elif [ "${1}" = "CTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/4i3pgy1a
elif [ "${1}" = "TCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/y0gmadlu
elif [ "${1}" = "TCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/e4fqv2jk
elif [ "${1}" = "TTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/w41a55s3
elif [ "${1}" = "TTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/hy2tkarj
fi
