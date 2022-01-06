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
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/gfboyuo2
elif [ "${1}" = "CTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/ia0ypryq
elif [ "${1}" = "CTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/nu5jk9zx
elif [ "${1}" = "TCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/nxue0t1g
elif [ "${1}" = "TTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/qad752hn
elif [ "${1}" = "TTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/55ay4rc2
fi
