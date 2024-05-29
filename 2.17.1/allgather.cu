#include <stdio.h>
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


static uint64_t getHostHash(const char* string) {
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i<maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

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
    int myRank, nRanks, localRank = 0;


    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    printf("nRanks: %d, myRank: %d, localRank: %d\n", nRanks, myRank, localRank);

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;
    int sendsize=size/nRanks, recvsize=sendsize*nRanks;


    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, sendsize * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, recvsize * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));


    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    double sum=0.0;
    for (int i=0; i<100; i++) {

    CUDACHECK(cudaDeviceSynchronize());
    gettimeofday(&start, NULL);


    NCCLCHECK(ncclAllGather((const void*)sendbuff, (void*)recvbuff, sendsize, ncclFloat, comm, s));


    CUDACHECK(cudaDeviceSynchronize());
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