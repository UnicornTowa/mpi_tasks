#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <random>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> dist(1, 65536);


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int n = 1000; n <= 10000000; n *= 10) {
        vector<int> vec;

        chrono::time_point<chrono::system_clock, chrono::nanoseconds> start_time;
        if (rank == 0) {
            for (auto i = 0; i < n; i++) {
                vec.push_back(dist(gen));
            }
            start_time = chrono::high_resolution_clock::now();
        }

        int local_size = n / size;
        vector<int> local_vec(local_size);

        MPI_Scatter(vec.data(), local_size, MPI_INT, local_vec.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

        int local_min = 65537;
        for (auto i: local_vec) {
            if (i < local_min) {
                local_min = i;
            }
        }

        int global_min;
        MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (n % size != 0) {
                for (int i = size * local_size; i < n; i++) {
                    if (vec[i] < global_min) {
                        global_min = vec[i];
                    }
                }
            }
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            cout << duration.count() << ",";
        }
    }
    MPI_Finalize();
    return 0;
}
