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
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/h14vdp1s
elif [ "${1}" = "CTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/ffzk05iv
elif [ "${1}" = "TCC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/ow74tdmn
elif [ "${1}" = "TTC" ];then
  wandb agent tatsukichishibuya/InvertibleTargetPropagation/unrbjoxh
fi
