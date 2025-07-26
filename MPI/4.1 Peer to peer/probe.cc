#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
    if (world_size != 2) {
        fprintf(stderr, "Error: This program requires exactly 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the current process

    int number_of_amount = 0;
    if (world_rank == 0) {
        int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];
        srand(time(NULL));
        number_of_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;
        MPI_Send(numbers, number_of_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent %d numbers to process %d.\n", world_rank, number_of_amount, 1);
    } else if (world_rank == 1) {
        MPI_Status status;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &number_of_amount);
        int* number_buffer = (int*)malloc(sizeof(int) * number_of_amount);
        MPI_Recv(number_buffer, number_of_amount, MPI_INT, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
            printf("1 dynamically received %d numbers from 0.\n",
                number_of_amount);
        free(number_buffer);
    }
    MPI_Finalize();
}
/******************************************************************
Process 0 sent 29 numbers to process 1.
1 dynamically received 29 numbers from 0.
*******************************************************************/