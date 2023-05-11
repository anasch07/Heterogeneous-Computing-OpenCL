#
# Matrix Multiplication Driver C = A * B
# Classic Sequential program on CPU
# without using OpenCL 
#                 
# A and B are constant matrices, square and the order is
# set as a constant =N. 
#
#
from helper import *
from definitions import *

import numpy
from time import time

# A[N][N], B[N][N], C[N][N]
N = 256;

# Number of elements in the matrix
size = N * N

# A matrix: filled with AVAL (value set in definitions.py)
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)

print ("\n===== Sequential, matrix mult (dot prod), order", N, "on host CPU ======\n")

for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    seq_mat_mul_sdot(N, h_A, h_B, h_C)

    run_time = time() - start_time
    results(N, run_time)
