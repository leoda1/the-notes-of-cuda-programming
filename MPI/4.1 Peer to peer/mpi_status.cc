/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-19
 * File Name   : mpi_status.cc
 * Description : This is a simple example of using MPI_Status to get
 *               additional information about a message received from
 *               another process.
 *****************************************************************/
#include <mpi.h>
#include <iostream>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
  
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
      fprintf(stderr, "Must use two processes for this example\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    const int MAX_NUMBERS = 100;
    int numbers[MAX_NUMBERS];
    int number_amount;
    if (world_rank == 0) {
      // Pick a random amount of integers to send to process one
      srand(time(NULL));
      number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;
      // Send the amount of integers to process one
      MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
      printf("0 sent %d numbers to 1\n", number_amount);
    } else if (world_rank == 1) {
      MPI_Status status;
      // Receive at most MAX_NUMBERS from process zero
      MPI_Recv(numbers, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      // After receiving the message, check the status to determine how many
      // numbers were actually received
      MPI_Get_count(&status, MPI_INT, &number_amount);
      // Print off the amount of numbers, and also print additional information
      // in the status object
      printf("1 received %d numbers from 0. Message source = %d, tag = %d\n",
             number_amount, status.MPI_SOURCE, status.MPI_TAG);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
/******************************************************************
0 sent 91 numbers to 1
0 sent 91 numbers to 1
*******************************************************************/
