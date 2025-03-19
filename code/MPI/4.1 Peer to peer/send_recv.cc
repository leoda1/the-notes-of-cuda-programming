/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-18
 * File Name   : send_recv.cc
 * Description : Send and receive data using MPI_Send and MPI_Recv
 *****************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int number;
    if (world_rank == 0) {
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1){
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n", number);
    }
    MPI_Finalize();
    return 0;
}
/******************************************************************
base) joker@joker-2 4.1 Send & Recv % mpirun -np 2 ./send_recv
Process 1 received number -1 from process 0
(base) joker@joker-2 4.1 Send & Recv % mpirun -np 1 ./send_recv
World size must be greater than 1 for ./send_recv
Abort(1) on node 0 (rank 0 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 1) - process 0
*******************************************************************/