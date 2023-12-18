#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

void My_MPI_Bcast(void * buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto iters = (int)floor(log2(size));
    for(int i = 0; i < iters; i++){
        if (rank < (int)(pow(2, i))){
            MPI_Send(buf, count, datatype, (int)pow(2, i) + rank, 0, comm);
        }
        else if (rank < (int)(pow(2, i + 1))){
            MPI_Recv(buf, count, datatype, rank - (int)pow(2, i), 0, comm, MPI_STATUS_IGNORE);
        }
    }
    auto received = (int)pow(2, iters);
    if (rank < size - received){
        MPI_Send(buf, count, datatype, received + rank, 0, comm);
    }
    else if (rank >= received){
        MPI_Recv(buf, count, datatype, rank - received, 0, comm, MPI_STATUS_IGNORE);
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int k = 100000; k >= 10; k /= 10) {
//    for (int k = 10; k >= 10; k /= 10) {
        string r(k, ' ');
        time_point<system_clock, nanoseconds>start_time;
        if (rank == 0) {
            r = string(k, 'a');
            start_time = high_resolution_clock::now();
        }
        MPI_Bcast((void *)r.data(), r.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
//        My_MPI_Bcast((void *)r.data(), r.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank == 0){
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            cout << duration.count() << ",";
        }
    }
    if (rank == 0) {
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}
