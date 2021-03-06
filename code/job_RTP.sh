#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=04:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "fashion" ];then
  wandb agent tatsukichishibuya/InvTP/kvpgwq6f
fi
