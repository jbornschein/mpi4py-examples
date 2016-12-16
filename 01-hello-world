#!/usr/bin/env python

from __future__ import print_function
from mpi4py import MPI


comm = MPI.COMM_WORLD

print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

comm.Barrier()   # wait for everybody to synchronize _here_

