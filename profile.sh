nvidia-smi --lock-gpu-clocks=1590
nvidia-smi --lock-memory-clocks=5000

echo "Matrix size 1024x1024"
echo "Kernel 1"
./build/runner 1 50 1024 1024 1024
echo "Kernel 2"
./build/runner 2 50 1024 1024 1024
echo "Kernel 3"
./build/runner 3 50 1024 1024 1024
echo "Kernel 4"
./build/runner 4 50 1024 1024 1024
echo "Kernel 5"
./build/runner 5 50 1024 1024 1024
echo "Kernel 6"
./build/runner 6 50 1024 1024 1024
echo "cuBLAS HGEMM"
./build/runner 99 50 1024 1024 1024
echo "------------------------------"

echo "Matrix size 2048x2048"
echo "Kernel 1"
./build/runner 1 50 2048 2048 2048
echo "Kernel 2"
./build/runner 2 50 2048 2048 2048
echo "Kernel 3"
./build/runner 3 50 2048 2048 2048
echo "Kernel 4"
./build/runner 4 50 2048 2048 2048
echo "Kernel 5"
./build/runner 5 50 2048 2048 2048
echo "Kernel 6"
./build/runner 6 50 2048 2048 2048
echo "cuBLAS HGEMM"
./build/runner 99 50 2048 2048 2048
echo "------------------------------"

echo "Matrix size 4096x4096"
echo "Kernel 1"
./build/runner 1 50 4096 4096 4096
echo "Kernel 2"
./build/runner 2 50 4096 4096 4096
echo "Kernel 3"
./build/runner 3 50 4096 4096 4096
echo "Kernel 4"
./build/runner 4 50 4096 4096 4096
echo "Kernel 5"
./build/runner 5 50 4096 4096 4096
echo "Kernel 6"
./build/runner 6 50 4096 4096 4096
echo "cuBLAS HGEMM"
./build/runner 99 50 4096 4096 4096
echo "------------------------------"


echo "Matrix size 8192x8192"
echo "Kernel 1"
./build/runner 1 50 8192 8192 8192
echo "Kernel 2"
./build/runner 2 50 8192 8192 8192
echo "Kernel 3"
./build/runner 3 50 8192 8192 8192
echo "Kernel 4"
./build/runner 4 50 8192 8192 8192
echo "Kernel 5"
./build/runner 5 50 8192 8192 8192
echo "Kernel 6"
./build/runner 6 50 8192 8192 8192
echo "cuBLAS HGEMM"
./build/runner 99 50 8192 8192 8192
echo "------------------------------"

echo "Matrix size 16384x16384"
echo "Kernel 1"
./build/runner 1 50 16384 16384 16384
echo "Kernel 2"
./build/runner 2 50 16384 16384 16384
echo "Kernel 3"
./build/runner 3 50 16384 16384 16384
echo "Kernel 4"
./build/runner 4 50 16384 16384 16384
echo "Kernel 5"
./build/runner 5 50 16384 16384 16384
echo "Kernel 6"
./build/runner 6 50 16384 16384 16384
echo "cuBLAS HGEMM"
./build/runner 99 50 16384 16384 16384
echo "------------------------------"

nvidia-smi --reset-gpu-clocks
nvidia-smi --reset-memory-clocks