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
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.04299251373155531 --epochs=200 --learning_rate=2.7985185386558067 --learning_rate_for_backward=0.02874860776145025 --refinement_iter=50 --stepsize=0.02 --type=TTT --log --agent
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.04684015427761784 --depth=4 --epochs=200 --hid_dim=1024 --in_dim=3072 --learning_rate=0.17762986867591446 --learning_rate_for_backward=0.003205928432704797 --problem=CIFAR10 --stepsize=0.02 --type=CCC --log --agent
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.03182888193163963 --depth=4 --epochs=200 --hid_dim=1024 --in_dim=3072 --learning_rate=1.6603198041448202 --learning_rate_for_backward=0.008906793194134368 --problem=CIFAR10 --refinement_iter=50 --stepsize=0.02 --type=TTT --log --agent
