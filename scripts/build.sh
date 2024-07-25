#!/bin/bash
set -e

if [ ! -d "kokkos" ]; then
    git clone https://github.com/kokkos/kokkos.git
fi
mkdir -p build && cd build

if [ "$1" == "--gpu" ] || [ "$1" == "-g" ]; then
    cmake .. -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DCMAKE_BUILD_TYPE=Release
else
    cmake .. -DKokkos_ENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
fi
make -j4

echo "Compilation finished. Run ./build/main to execute."
