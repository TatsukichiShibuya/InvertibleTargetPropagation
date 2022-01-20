#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=14:55:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
wandb agent tatsukichishibuya/InvertibleTargetPropagation/7th9fkpy
