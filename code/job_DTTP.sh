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
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.04324976373233741 --depth=4 --epochs=200 --hid_dim=1024 --in_dim=3072 --learning_rate=0.27598331991824104 --learning_rate_for_backward=0.003413192551819537 --problem=CIFAR10 --stepsize=0.02 --type=CCC --log --agent
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.03054574476610649 --depth=4 --epochs=200 --hid_dim=1024 --in_dim=3072 --learning_rate=1.4215620527633597 --learning_rate_for_backward=0.007174953032885884 --problem=CIFAR10 --refinement_iter=50 --stepsize=0.02 --type=TTT --log --agent
