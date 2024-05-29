#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N prof_allreduce

. /etc/profile.d/modules.sh
module load gcc/8.3.0 intel cuda/11.2.146 openmpi nccl/2.8.4

nvcc allreduce.cu -lmpi -lnccl

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="info.%h.%p.txt"
export NCCL_BUFFSIZE=1048576
export NCCL_TOPO_DUMP_FILE="sys_topo.txt"
mpirun -x LD_LIBRARY_PATH -x PATH -n 8 -npernode 4 --bind-to none ./a.out
