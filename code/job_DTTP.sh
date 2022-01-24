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
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.04845901001187481 --epochs=200 --learning_rate=1.8346872617067749 --learning_rate_for_backward=0.02936765465726871 --refinement_iter=50 --stepsize=0.02 --type=TTT --log --agent
python Main.py --algorithm=DTTP --activation_function=tanh --b_epochs=3 --b_sigma=0.03455327947447475 --epochs=200 --learning_rate=2.958866958479571 --learning_rate_for_backward=0.02171245839470964 --stepsize=0.02 --type=CCC --log --agent
