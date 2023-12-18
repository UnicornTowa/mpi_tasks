#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>
#include <vector>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int n = 1024 * 1024; n >= 1; n/=2){

        auto num_messages = 1000;
        MPI_Request request[2 * num_messages];
        MPI_Status status[2 * num_messages];
        vector<string> messages(num_messages, string(n, ' '));

        time_point<system_clock, nanoseconds> start_time;
        if (rank == 0){
            start_time = high_resolution_clock::now();
        }

         for(int count = 1; count <= num_messages; count++) {
            if (rank == 0) {
                string send0(n, 'a');
                MPI_Isend(send0.data(), static_cast<int>(send0.size()), MPI_CHAR, 1, 0,
                          MPI_COMM_WORLD, &request[count - 1]);

//                string recv0(n, ' ');
                MPI_Irecv(messages[count - 1].data(), static_cast<int>(messages[count - 1].size()), MPI_CHAR, 1, 0,
                          MPI_COMM_WORLD, &request[count - 1 + num_messages]);
//                cout << "rank 0 received " << recv0 << endl;
            }
            if (rank == 1) {
//                string recv1(n, ' ');
                MPI_Irecv(messages[count - 1].data(), static_cast<int>(messages[count - 1].size()), MPI_CHAR, 0, 0,
                          MPI_COMM_WORLD, &request[count - 1]);
//                cout << "rank 1 received " << recv1 << endl;

                string send1(n, 'b');
                MPI_Isend(send1.data(), static_cast<int>(send1.size()), MPI_CHAR, 0, 0,
                          MPI_COMM_WORLD, &request[count - 1 + num_messages]);
            }
        }
        MPI_Waitall(2 * num_messages, request, status);

//         for(auto &i : messages){
//             cout << i << endl;
//         }

         if (rank == 0){
             auto end_time = high_resolution_clock::now();
             auto duration = duration_cast<microseconds>(end_time - start_time);
             cout << duration.count() << ",";
         }
    }
    MPI_Finalize();
    return 0;
}
