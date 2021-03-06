#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N bidirec_comm_test

. /etc/profile.d/modules.sh
module load cuda/10.2.89 openmpi nccl


#NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=1048576
mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 2 --bind-to none ./a.out
