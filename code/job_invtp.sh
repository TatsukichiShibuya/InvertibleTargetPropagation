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
if [ "${1}" = "invCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/d7xj7wji
elif [ "${1}" = "invCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/i37ckmdn
elif [ "${1}" = "invTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/x6ubkt4x
elif [ "${1}" = "invTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/yyxhuz4b
fi
