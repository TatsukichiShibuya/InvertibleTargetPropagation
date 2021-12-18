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
if [ "${1}" = "eye-fg" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/6fys9nt6
elif [ "${1}" = "eye-gf" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/rx8lgs60
elif [ "${1}" = "fg-fg" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/tao3gnle
elif [ "${1}" = "gf-gf" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/lu49kejg
elif [ "${1}" = "inv-fg" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/2j1tq5sy
elif [ "${1}" = "inv-gf" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/pefur76v
else
  echo "error"
fi
