#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <vector>

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

void My_MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto temp_vec = vector<int>(size * recvcount, 0);
    if (rank == root){
        auto ptr = static_cast<const int *>(sendbuf);
        temp_vec = vector<int>(ptr, ptr + size * sendcount);
    }
    auto iters = (int)floor(log2(size));
    for(int i = 0; i < iters; i++){
        if (rank < (int)(pow(2, i))){
            MPI_Send(temp_vec.data(), size * recvcount, sendtype, (int)pow(2, i) + rank, 0, comm);
        }
        else if (rank < (int)(pow(2, i + 1))){
            MPI_Recv(temp_vec.data(), size * recvcount, recvtype, rank - (int)pow(2, i), 0, comm, MPI_STATUS_IGNORE);
        }
    }
    auto received = (int)pow(2, iters);
    if (rank < size - received){
        MPI_Send(temp_vec.data(), size * recvcount, sendtype, received + rank, 0, comm);
    }
    else if (rank >= received){
        MPI_Recv(temp_vec.data(), size * recvcount, recvtype, rank - received, 0, comm, MPI_STATUS_IGNORE);
    }
    auto my_data = vector<int>(temp_vec.begin() + rank * recvcount, temp_vec.begin() + recvcount * (rank + 1));
    auto ptr = static_cast<int *>(recvbuf);
    for (int i = 0; i < recvcount; i++){
        ptr[i] = my_data[i];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int k = 600000; k >= 60; k /= 10) {
//    for (int k = 10; k >= 10; k /= 10) {
        vector<int> vec;
        vector<int> local_vec(k / size, 0);
        time_point<system_clock, nanoseconds>start_time;
        if (rank == 0) {
            for(int i = 0; i < k; i++){
                vec.push_back(i);
            }
            start_time = high_resolution_clock::now();
        }
        My_MPI_Scatter((void *)vec.data(), vec.size() / size, MPI_INT, local_vec.data(), local_vec.size(), MPI_INT, 0, MPI_COMM_WORLD);
//        MPI_Scatter((void *)vec.data(), vec.size() / size, MPI_INT, local_vec.data(), local_vec.size(), MPI_INT, 0, MPI_COMM_WORLD);
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
