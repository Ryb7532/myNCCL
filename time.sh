#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N p2p_multinode

. /etc/profile.d/modules.sh
module load gcc/8.3.0 intel cuda/11.2.146 openmpi nccl

#NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=524288
for i in 4 16 64 256 
do
    echo "data: ${i}K"
    # mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 2 --bind-to none ./a.out $i
    mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 2 --bind-to none ./a.out $i
    echo
done

for i in 1 4 16 64 256 1024
do
    echo "data: ${i}M"
    # mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 2 --bind-to none ./a.out $i
    size=`expr ${i} \* 1024`
    mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 2 --bind-to none ./a.out $size
    echo
done
