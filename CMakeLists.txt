cmake_minimum_required(VERSION 3.10)
project(Project)

add_subdirectory(kokkos)
add_executable(main main.cpp)
target_link_libraries(main Kokkos::kokkos)
