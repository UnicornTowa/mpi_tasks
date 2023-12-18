#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int n = 1024*1024; n >= 1; n/=2){
        time_point<system_clock, nanoseconds> start_time;
        if (rank == 0){
            start_time = high_resolution_clock::now();
        }
        for(int count = 1; count <= 1000; count++) {
            if (rank == 0) {
                string send0(n, 'a');
//                MPI_Rsend(send0.data(), static_cast<int>(send0.size()), MPI_CHAR, 1, 0, MPI_COMM_WORLD);

                string recv0(n, ' ');
//                MPI_Recv(recv0.data(), static_cast<int>(recv0.size()), MPI_CHAR, 1, 0, MPI_COMM_WORLD,
//                         MPI_STATUS_IGNORE);

                MPI_Sendrecv(send0.data(), static_cast<int>(send0.size()), MPI_CHAR, 1, 0, recv0.data(),
                             static_cast<int>(recv0.size()), MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                cout << "rank 0 received " << recv0 << endl;
            }
            if (rank == 1) {
                string recv1(n, ' ');
//                MPI_Recv(recv1.data(), static_cast<int>(recv1.size()), MPI_CHAR, 0, 0, MPI_COMM_WORLD,
//                         MPI_STATUS_IGNORE);
//                cout << "rank 1 received " << recv1 << endl;

                string send1(n, 'b');
//                MPI_Rsend(send1.data(), static_cast<int>(send1.size()), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                MPI_Sendrecv(send1.data(), static_cast<int>(send1.size()), MPI_CHAR, 0, 0, recv1.data(),
                             static_cast<int>(recv1.size()), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (rank == 0){
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            cout << duration.count() << ",";
        }
    }
    MPI_Finalize();
    return 0;
}
