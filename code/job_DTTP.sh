#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=6:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
python Main.py --algorithm=BP --epochs=200 --learning_rate=0.09818910925650234 --log --agent
python Main.py --algorithm=BP --epochs=200 --hid_dim=1024 --in_dim=3072 --learning_rate=0.023236041887720604 --problem=CIFAR10 --log --agent
