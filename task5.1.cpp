#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <random>

using namespace std;
using namespace chrono;

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<short> dist(-100, 100);



typedef short ULL;
#define MPI_UNSIGNED_LONG_LONG MPI_SHORT

void transpose(vector<vector<ULL>>& matrix){
    auto n = matrix.size();
    for (int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}

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
    vector<int> n_list = {480, 240, 120};
    for (auto n : n_list) {

        vector<vector<ULL>> matrix_a = vector<vector<ULL>>(n, vector<ULL>(n, 0));
        vector<vector<ULL>> matrix_b = vector<vector<ULL>>(n, vector<ULL>(n, 0));

        time_point<system_clock, nanoseconds> start_time;
        if (rank == 0) {
            for(int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    matrix_a[i][j] = dist(gen);
                    matrix_b[i][j] = dist(gen);
                }
            }
            start_time = high_resolution_clock::now();
            transpose(matrix_b);
        }
//        cout << "generating done" << endl;
        int local_size = n / size;
        vector<vector<ULL>> local_rows_const(vector<vector<ULL>>(local_size, vector<ULL>(n, 0)));
        vector<vector<ULL>> local_rows_dyn(vector<vector<ULL>>(local_size, vector<ULL>(n, 0)));
        MPI_Request request[2*n];
        MPI_Status status[2*n];
//        cout << "root starting sending" << endl;
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < local_size; j++) {
                    MPI_Isend(matrix_a[i * local_size + j].data(), n,
                              MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[i * local_size + j]);

                    MPI_Isend(matrix_b[i * local_size + j].data(), n,
                              MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, &request[n + i * local_size + j]);
                }
            }
//            cout << "waiting all" << endl;
            MPI_Waitall(2 * n, request, status);
//            cout << "done" << endl;

        }
//        cout << "root sent data" << endl;
        for(int j = 0; j < local_size; j++){
            MPI_Recv(local_rows_const[j].data(), n, MPI_UNSIGNED_LONG_LONG, 0, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_rows_dyn[j].data(), n, MPI_UNSIGNED_LONG_LONG, 0, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

//        cout << "distribution done" << endl;

        vector<vector<ULL>> local_res(local_size, (vector<ULL>(n, 0)));

        for(int i = 0; i < size; i++){
            for(int j = 0; j < local_size; j++){
                for(int k = 0; k < local_size; k++){
                    local_res[j][(rank + i) * local_size % n + k] = dot_product(local_rows_const[j], local_rows_dyn[k]);
                }
            }
            for(int j = 0; j < local_size; j++){
                MPI_Send(local_rows_dyn[j].data(), n, MPI_UNSIGNED_LONG_LONG, (rank - 1 >= 0 ? rank - 1 : size - 1), 0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Recv(local_rows_dyn[j].data(), n, MPI_UNSIGNED_LONG_LONG, (rank + 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

//        cout << "local res ready" << endl;

        vector<vector<ULL>> global_res(n, (vector<ULL>(n, 0)));


        for(int j = 0; j < local_size; j++){
            MPI_Send(local_res[j].data(), int(local_res[j].size()),
                      MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < local_size; j++) {
                    MPI_Recv(global_res[i * local_size + j].data(),
                              static_cast<int>(global_res[i * local_size + j].size()), MPI_UNSIGNED_LONG_LONG, i, 0,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        if (rank == 0) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            cout << duration.count() << ",";
        }
    }
    MPI_Finalize();
    return 0;
}