#!/bin/sh
#$ -cwd
#$ -l f_node=8
#$ -l h_rt=00:10:00
#$ -N allreduce

. /etc/profile.d/modules.sh
module load gcc/8.3.0 intel cuda/11.2.146 openmpi nccl

#export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=524288

mpirun -x LD_LIBRARY_PATH -x PATH -np 8 -npernode 1 --bind-to none ./build/all_reduce_perf -b 4K -e 4096M -f 4 -g 4
