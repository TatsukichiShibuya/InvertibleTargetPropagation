#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -o output/o.$JOB_ID
wandb agent tatsukichishibuya/InvertibleTargetPropagation/xeb4wgjp
