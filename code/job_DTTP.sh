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
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/t6ghjn64
elif [ "${1}" = "CCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/p24cllj1
elif [ "${1}" = "CTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/ltp2e5j4
elif [ "${1}" = "CTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/tmpsr9zk
elif [ "${1}" = "TCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/suabwuts
elif [ "${1}" = "TCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/gps4td84
elif [ "${1}" = "TTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/s0ryywq1
elif [ "${1}" = "TTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/9a04ttm5
fi
