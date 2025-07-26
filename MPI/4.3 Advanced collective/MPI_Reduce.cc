#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>

float* create_rand_nums (int num_elements) {
    float* rand_nums = (float*)malloc(num_elements * sizeof(float));
    assert(rand_nums != NULL);
    for (int i = 0; i < num_elements; i ++) {
        rand_nums[i] = ((float) rand() / RAND_MAX);
    }
    return rand_nums;
}

int main (int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: avg num_elements_per_proc\n");
        exit(1);
    }

    int num_ele_per_proc = atoi(argv[1]);
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(NULL) * world_rank);
    float *rand_nums = NULL;
    rand_nums = create_rand_nums(num_ele_per_proc);

    float local_sum = 0;
    for (int i = 0; i < num_ele_per_proc; i ++) {
        local_sum += rand_nums[i];
    }
    printf("Local sum for process %d - %f, avg = %f\n",
        world_rank, local_sum, local_sum / num_ele_per_proc);
    
    float global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Total sum = %f, avg = %f\n", global_sum, global_sum / (world_size * num_ele_per_proc));
    }
    
    // Clean up
    free(rand_nums);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
/******************************************************************
(base) joker@joker-2 4.3 Advanced collective % mpic++ MPI_Reduce_exp1.cc -o MPI_Reduce_exp1
(base) joker@joker-2 4.3 Advanced collective % mpirun -np 4 ./MPI_Reduce_exp1 100
Local sum for process 1 - 48.185650, avg = 0.481856
Local sum for process 2 - 52.371292, avg = 0.523713
Local sum for process 3 - 52.872005, avg = 0.528720
Local sum for process 0 - 51.385098, avg = 0.513851
Total sum = 204.814056, avg = 0.512035
*******************************************************************/