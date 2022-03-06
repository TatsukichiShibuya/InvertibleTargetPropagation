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
  wandb agent tatsukichishibuya/InvTP/tvj6uyvj
elif [ "${1}" = "fashion" ];then
  wandb agent tatsukichishibuya/InvTP/67ewkq6e
elif [ "${1}" = "mnist" ];then
  wandb agent tatsukichishibuya/InvTP/xmopu8e3
elif [ "${1}" = "augmented-cifar" ];then
  wandb agent tatsukichishibuya/InvTP/sd9m7ua9
elif [ "${1}" = "augmented-fashion" ];then
  wandb agent tatsukichishibuya/InvTP/adrq8jw9
elif [ "${1}" = "augmented-mnist" ];then
  wandb agent tatsukichishibuya/InvTP/i47lv9iy
fi
