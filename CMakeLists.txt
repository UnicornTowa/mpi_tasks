cmake_minimum_required(VERSION 3.22)
project(mpi_hw)

set(CMAKE_CXX_STANDARD 17)


find_package(MPI REQUIRED)

add_executable(mpi_hw main.cpp)

target_link_libraries(mpi_hw PRIVATE MPI::MPI_CXX)