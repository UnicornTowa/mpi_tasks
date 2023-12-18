#include <iostream>
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <vector>

using namespace std;
using namespace chrono;

void My_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto ptr_send = static_cast<const int *>(sendbuf);
    auto in_vec = vector<int>(ptr_send, ptr_send + sendcount);
    auto temp_vec = vector<int>(size * recvcount, 0);
    copy(in_vec.begin(), in_vec.end(), temp_vec.begin() + rank * sendcount);
    auto iters = (int)floor(log2(size));
    for(int i = 0; i < iters; i++){
        auto pack = (int)pow(2, i);
        if (rank % (2 * pack) == 0 && rank + pack < size){
            MPI_Recv(&*(temp_vec.begin() + (pack + rank) * sendcount), recvcount * (pack), sendtype, pack + rank, 0, comm, MPI_STATUS_IGNORE);

        }
        else if (rank % (2 * pack) == (int)pow(2, i)){
            MPI_Send(&*(temp_vec.begin() + rank * sendcount), recvcount * (pack), recvtype, rank - pack, 0, comm);
        }
    }
    auto res_rank = (int)pow(2, iters);
    if (rank == res_rank){
        MPI_Send(&*(temp_vec.begin() + (res_rank) * sendcount), recvcount * (size - res_rank), recvtype, 0, 0, comm);
    }
    if (rank == 0 && iters != (int) ceil(log2(size))){
        MPI_Recv(&*(temp_vec.begin() + (res_rank) * sendcount), recvcount * (size - res_rank), recvtype, res_rank, 0, comm, MPI_STATUS_IGNORE);
    }

    auto ptr_recv = static_cast<int *>(recvbuf);
    for (int i = 0; i < recvcount * size; i++){
        ptr_recv[i] = temp_vec[i];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    for (int k = 600000; k >= 60; k /= 10) {
    for (int k = 100; k >= 100; k /= 10) {
        vector<int> vec(k, 0);
        vector<int> local_vec;
        time_point<system_clock, nanoseconds>start_time;
        for (int i = 0; i < k / size; i++){
            local_vec.push_back(k * rank / size + i);
        }
        if (rank == 0) {
            start_time = high_resolution_clock::now();
        }
        My_MPI_Gather(local_vec.data(), local_vec.size(), MPI_INT, vec.data(), k / size, MPI_INT, 0, MPI_COMM_WORLD);
//        MPI_Gather(local_vec.data(), local_vec.size(), MPI_INT, vec.data(), k / size, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0){
            for (auto i : vec)
                cout << i << ", ";
            cout << endl;
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
