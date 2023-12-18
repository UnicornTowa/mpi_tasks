#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <random>
#include <unistd.h>
#include <string>

using namespace std;
using namespace chrono;

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<short> dist(-100, 100);


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for(int nbytes = 1; nbytes <= 100; nbytes *= 10){
        for(int nmessages = 1; nmessages <= 1000; nmessages *= 10){
            for(int sleep_time = 10; sleep_time <= 1000; sleep_time *= 10) {
                time_point<system_clock, nanoseconds> start_time;
                if (rank == 0) {
                    start_time = high_resolution_clock::now();
                }
                for (int j = 0; j < 1000 / size; j++) {
                    usleep(sleep_time);
                    string s(nbytes, 'a');
                    string r(nbytes, ' ');
                    for (int i = 0; i < nmessages; i++) {

                        MPI_Send(s.data(), s.size(), MPI_CHAR, (rank + 1) % size, 0, MPI_COMM_WORLD);
                        MPI_Recv((void *) r.data(), nbytes, MPI_CHAR, (rank > 0 ? rank - 1 : size - 1), 0,
                                 MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    }
                }
                if (rank == 0) {
                    auto end_time = high_resolution_clock::now();
                    auto duration = duration_cast<microseconds>(end_time - start_time);
                    cout << size << "," << nbytes << "," << nmessages << "," << sleep_time << "," << duration.count() << endl;
                }
            }
        }
    }


    MPI_Finalize();
    return 0;
}