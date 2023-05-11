#
# Matrix Multiplication Driver C = A * B
# Using OpenCL (Heteregeneous Computing) U
# To run on GPU or Multicore CPU 
#                 
# A and B are constant matrices, square and the order is
# set as a constant, ORDER (see definitions.py). 
#

from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time

# A[N][N], B[N][N], C[N][N]
N = 1024;

# Number of elements in the matrix
size = N * N
localsize=16

# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)


# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item
#--------------------------------------------------------------------------------

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)


kernelsource = open("C_row.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])
print ("\n===== OpenCL, matrix mult, C column per work item, order", N, "======\n")
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    mmul(queue, (N,), (localsize,), N, d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, run_time)

#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... C row per work item, A row in pivate memory
#--------------------------------------------------------------------------------


kernelsource = open("C_row.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])
print ("\n===== OpenCL, matrix mult, C row per work item, order", N, "======\n")
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    mmul(queue, (N,), None, N, d_a, d_b, d_c)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, run_time)