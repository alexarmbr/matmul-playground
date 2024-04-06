FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y ca-certificates gpg wget && \
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update

RUN apt-get update && apt-get install -y cmake libgtest-dev git python3 python3-pip
RUN pip install cuda-python
RUN mkdir /tmp/cutlass && \
    git clone https://github.com/NVIDIA/cutlass.git /tmp/cutlass && \
    cd /tmp/cutlass && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCUTLASS_NVCC_ARCHS=75 -DCUTLASS_LIBRARY_KERNELS=gemm && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/cutlass