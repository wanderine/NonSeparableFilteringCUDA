#!/bin/bash 

CUDAHELPERPATH=/usr/local/cuda-7.5/samples/common/inc
CUDALIBPATH=/usr/local/cuda-7.5/lib64/
CUDABINPATH=/usr/local/cuda-7.5/bin/

${CUDABINPATH}/nvcc -lib -I${CUDAHELPERPATH} -lcufft -L${CUDALIBPATH} -O2 -m64 filtering.cu -o libFilteringCUDA.a


