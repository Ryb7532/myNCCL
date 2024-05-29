#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N allreduce

. /etc/profile.d/modules.sh
module load gcc/8.3.0 cuda openmpi nccl

nvcc allreduce.cu -lmpi -lnccl

#export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=1048576
for i in 32 64 128 256 512 1024 4096
do
    echo "data: ${i}K"
    # mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 2 --bind-to none ./a.out $i
    mpirun -x LD_LIBRARY_PATH -x PATH -n 8 -npernode 4 --bind-to none ./a.out $i
    echo
done

#export NCCL_DEBUG=INFO
for i in 16 64 256 1024
do
    echo "data: ${i}M"
    size=`expr $i \* 1024`
    mpirun -x LD_LIBRARY_PATH -x PATH -n 8 -npernode 4 --bind-to none ./a.out $size
    echo
done
