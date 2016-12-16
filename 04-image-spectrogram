#!/usr/bin/env python 

"""

How to run:

   mpirun -np <NUM> ./image-spectrogram <IMAGES.h5>


This example computes the 2D-FFT of every image inside <IMAGES.h5>, 
Summs the absolute value of their spectrogram and finally displays
the result in log-scale.
"""

from __future__ import division

import sys
import tables
import pylab
import numpy as np
from numpy.fft import fft2, fftshift
from mpi4py import MPI
from parutils import pprint

#=============================================================================
# Main

comm = MPI.COMM_WORLD

in_fname = sys.argv[-1]

try:
    h5in  = tables.openFile(in_fname, 'r')
except:
    pprint("Error: Could not open file %s" % in_fname)
    exit(1)

images = h5in.root.images
image_count, height, width = images.shape
image_count = min(image_count, 800)

pprint("============================================================================")
pprint(" Running %d parallel MPI processes" % comm.size)
pprint(" Reading images from '%s'" % in_fname)
pprint(" Processing %d images of size %d x %d" % (image_count, width, height))

# Distribute workload so that each MPI process analyzes image number i, where
#  i % comm.size == comm.rank. 
#
# For example if comm.size == 4:
#   rank 0: 0, 4, 8, ...
#   rank 1: 1, 5, 9, ...
#   rank 2: 2, 6, 10, ...
#   rank 3: 3, 7, 11, ...

comm.Barrier()                    ### Start stopwatch ###
t_start = MPI.Wtime()

my_spec = np.zeros( (height, width) )
for i in range(comm.rank, image_count, comm.size):
    img  = images[i]            # Load image from HDF file
    img_ = fft2(img)            # 2D FFT
    my_spec += np.abs(img_)     # Sum absolute value into partial spectrogram

my_spec /= image_count

# Now reduce the partial spectrograms into *spec* by summing 
# them all together. The result is only avalable at rank 0.
# If you want the result to be availabe on all processes, use
# Allreduce(...)
spec = np.zeros_like(my_spec)
comm.Reduce(
    [my_spec, MPI.DOUBLE],
    [spec, MPI.DOUBLE],
    op=MPI.SUM,
    root=0
)

comm.Barrier()
t_diff = MPI.Wtime()-t_start      ### Stop stopwatch ###

h5in.close()

pprint(
    " Analyzed %d images in %5.2f seconds: %4.2f images per second" % 
        (image_count, t_diff, image_count/t_diff) 
)
pprint("============================================================================")

# Now rank 0 outputs the resulting spectrogram.
# Either onto the screen or into a image file.
if comm.rank == 0:
    spec = fftshift(spec)
    pylab.imshow(np.log(spec))
    pylab.show()

comm.Barrier()

