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
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/qc5ahns2
elif [ "${1}" = "CCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/2z284idb
elif [ "${1}" = "CTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/osphe206
elif [ "${1}" = "CTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/vkkzie3q
elif [ "${1}" = "TCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/ot4dqtav
elif [ "${1}" = "TCT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/t9lawsqh
elif [ "${1}" = "TTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/uvkckpcd
elif [ "${1}" = "TTT" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/km8zrb82
elif [ "${1}" = "noref" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/zojyslxl
fi
