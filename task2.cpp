#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <random>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> dist_mil(0, 1048575);

typedef unsigned long long ULL;
ULL dot_product(const vector<ULL>& v1, const vector<ULL>& v2){
    ULL result = 0;
    for(int i = 0; i < v1.size(); i++){
        result += v1[i]*v2[i];
    }
    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //for (int n = 1000; n <= 1000000; n *= 10) {
    for (int n = 1000000; n >= 1000; n /= 10) {
        vector<ULL> vec1;
        vector<ULL> vec2;

        chrono::time_point<chrono::system_clock, chrono::nanoseconds> start_time;
        if (rank == 0) {
            for (auto i = 0; i < n; i++) {
                vec1.push_back(dist_mil(gen));
                vec2.push_back(dist_mil(gen));
            }
            start_time = chrono::high_resolution_clock::now();
        }

        int local_size = n / size;
        vector<ULL> local_vec1(local_size);
        vector<ULL> local_vec2(local_size);
        MPI_Scatter(vec1.data(), local_size, MPI_UNSIGNED_LONG_LONG, local_vec1.data(), local_size, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Scatter(vec2.data(), local_size, MPI_UNSIGNED_LONG_LONG, local_vec2.data(), local_size, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        ULL local_res = dot_product(local_vec1, local_vec2);

        ULL global_res;
        MPI_Reduce(&local_res, &global_res, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (n % size != 0) {
                for (int i = size * local_size; i < n; i++) {
                    global_res += dot_product(vector<ULL>(vec1.begin() + i, vec1.end()), vector<ULL>(vec2.begin() + i, vec2.end()));
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
//
// Created by tosha on 07/12/2023.
//
