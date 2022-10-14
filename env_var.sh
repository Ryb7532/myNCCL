#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N test_gpudirect

. /etc/profile.d/modules.sh
module load gcc/8.3.0 intel cuda/11.2.146 openmpi nccl

export NCCL_DEBUG=INFO
# export NCCL_BUFFSIZE=524288
# export NCCL_TOPO_DUMP_FILE=system.txt

echo "start"
mpirun -x LD_LIBRARY_PATH -x PATH -x PSM2_CUDA=1 -x PSM2_GPUDIRECT=1 -n 8 -npernode 4 --bind-to none ./a.out 131072
# mpirun -x LD_LIBRARY_PATH -x PATH -n 8 -npernode 4 --bind-to none ./wrap.sh ./a.out 131072
