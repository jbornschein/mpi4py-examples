#!/usr/bin/env python 

"""

Demonstrating a MPI parallel Matrix-Vector Multiplication.

This code will run *iter* iterations of

  v(t+1) = M * v(t)

where v is a vector of length *size* and M a dense size*size
matrix. *size* must be an integer multiple of comm.size.

v is initialized to be zero except of v[0] = 1.0
M is a "off-by-one" diagonal matrix M[i, i+1] = 1.0

In effect, after *iter* iterations, the vector v should look like
v[iter] = 1. (all others zero).


In this example every MPI process is responsible for calculating a 
different portion of v. Every process only knows the stripe of M, that 
is relevant for it's calculation. At the end of every iteration, 
Allgather is used to distribute the partial vectors v to all other 
processes.

"""

from __future__ import division

import numpy as np
from numpy.fft import fft2, ifft2
from math import ceil, fabs
from mpi4py import MPI
from parutils import pprint

#=============================================================================
# Main

size = 10000           # lengt of vector v
iter = 20              # number of iterations to run

comm = MPI.COMM_WORLD

pprint("============================================================================")
pprint(" Running %d parallel MPI processes" % comm.size)

my_size = size // comm.size     # Every process computes a vector of lenth *my_size*
size = comm.size*my_size        # Make sure size is a integer multiple of comm.size
my_offset = comm.rank*my_size

# This is the complete vector
vec = np.zeros(size)            # Every element zero...
vec[0] = 1.0                    #  ... besides vec[0]

# Create my (local) slice of the matrix
my_M = np.zeros((my_size, size))
for i in xrange(my_size):
    j = (my_offset+i-1) % size
    my_M[i,j] = 1.0

comm.Barrier()                    ### Start stopwatch ###
t_start = MPI.Wtime()

for t in xrange(iter):
    my_new_vec = np.inner(my_M, vec)
    
    comm.Allgather(
        [my_new_vec, MPI.DOUBLE], 
        [vec, MPI.DOUBLE] 
    )

comm.Barrier()
t_diff = MPI.Wtime() - t_start    ### Stop stopwatch ###

if fabs(vec[iter]-1.0) > 0.01:
    pprint("!! Error: Wrong result!")

pprint(" %d iterations of size %d in %5.2fs: %5.2f iterations per second" %
    (iter, size, t_diff, iter/t_diff) 
)
pprint("============================================================================")

