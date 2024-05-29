#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N res_reducescatter

. /etc/profile.d/modules.sh
module load cuda openmpi nccl

#export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=524288

# mpirun -x LD_LIBRARY_PATH -x PATH -np 2 -npernode 1 --bind-to none ./build/all_reduce_perf -b 4K -e 4096M -f 4 -g 4
# mpirun -x LD_LIBRARY_PATH -x PATH -np 2 -npernode 1 --bind-to none ./build/all_gather_perf -b 4K -e 4096M -f 4 -g 4
mpirun -x LD_LIBRARY_PATH -x PATH -np 2 -npernode 1 --bind-to none ./build/reduce_scatter_perf -b 4K -e 4096M -f 4 -g 4
