#
# Matrix Multiplication Driver C = A * B
# Using OpenCL (Heteregeneous Computing) U
# To run on GPU or Multicore CPU 
#                 
# A and B are constant matrices, square and the order is
# set as a constant, ORDER (see definitions.py). 
#
#

from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time

# A[N][N], B[N][N], C[N][N]
N = 1024

# Number of elements in the matrix
size = N * N
ilocalsize = 32
jlocalsize = 32

# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix0
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)

# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Instantiate a Queue with profiling (timing) enabled

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive
# Naive : Ecah Work item computes a Matrix Element
#--------------------------------------------------------------------------------

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

#kernelsource = open("C_elem.cl").read()

kernelsource = open("C_block_form.c").read()

program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None,None,None])

print ("\n===== OpenCL, matrix mult, C(i,j) per work item, order", N, "======\n")

print ("\n ==== Execution for localsize : ", ilocalsize, " * ", jlocalsize," === \n" )

# Do the multiplication COUNT times
total_time = 0.00
localmemA = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * 1024)
localmemB = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * 1024)


for i in range(COUNT):
    h_C.fill(0.0)
    gpu_start_time = time()  # Get the GPU start time
    event = mmul(queue, (N, N), (ilocalsize,jlocalsize), N, d_a, d_b, d_c,localmemA,localmemB)
    event.wait()  # Wait until the event finishes
    elapsed = 1e-9*(event.profile.end - event.profile.start)  # Calculate the time it took to execute the kernel
    print("GPU Kernel Time: {0} s".format(elapsed))  # Print the time it took to execute the kernel
    total_time=total_time+elapsed
       #queue.finish()
    

print ("\n== AVerage Performance for  ", COUNT, "  runs=====\n")
results(N, total_time)
cl.enqueue_copy(queue, h_C, d_c)
print (h_C[0])