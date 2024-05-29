#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "mpi.h"
#include <sys/time.h>


#define MPICHECK(cmd) do {                                  \
    int e = cmd;                                            \
    if (e != MPI_SUCCESS) {                                 \
        printf("Failed: MPI error %s:%d '%d'\n",            \
                __FILE__, __LINE__, e);                     \
    }                                                       \
} while (0)


#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
        printf("Failed: CUDA error %s:%d '%s'\n",           \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                       \
} while (0)


#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                   \
    if (r != ncclSuccess) {                                 \
        printf("Failed: NCCL error %s:%d '%s'\n",           \
                __FILE__, __LINE__, ncclGetErrorString(r)); \
    }                                                       \
} while (0)


double get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000000
            + (end->tv_usec - begin->tv_usec);
}


int main(int argc, char* argv[]) {
    int size = 32*1024*1024;
    if (argc>=2) {
       size = atoi(argv[1])*1024;
    }

    struct timeval start, end;
    int myRank, nRanks;
    long localRank;

    localRank = strtol(getenv("OMPI_COMM_WORLD_LOCAL_RANK"), NULL, 10);
    CUDACHECK(cudaSetDevice(localRank));

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    int root = 4;

    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


    if (myRank == root) {
        CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float) * root));
        CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float) * root));
    } else {
        CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
        CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    }
    CUDACHECK(cudaStreamCreate(&s));


    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    double sum=0.0;
    for (int i=0; i<100; i++) {

    CUDACHECK(cudaStreamSynchronize(s));
    gettimeofday(&start, NULL);

    NCCLCHECK(ncclGroupStart());
    if (myRank == root) {
        size_t offset = size;
        for (int r=0; r<root; r++) {
            ncclRecv(recvbuff+r*offset, size, ncclFloat, r, comm, s);
            ncclSend(sendbuff+r*offset, size, ncclFloat, r, comm, s);
        }
    } else if (myRank < root) {
        ncclSend(sendbuff, size, ncclFloat, root, comm, s);
        ncclRecv(recvbuff, size, ncclFloat, root, comm, s);
    }
    NCCLCHECK(ncclGroupEnd());


    CUDACHECK(cudaStreamSynchronize(s));
    gettimeofday(&end, NULL);

    {
	double us;
      	us = get_elapsed_time(&start, &end);
	sum += us/1000.0;
    }

    }

    printf("(Rank %d) time: %.3lf ms\n", myRank, sum/100);

    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));


    ncclCommDestroy(comm);


    MPICHECK(MPI_Finalize());


    // printf("[MPI Rank %d] Succcess\n", myRank);
    return 0;
}