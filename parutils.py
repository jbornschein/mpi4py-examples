"""
  Some utility functions useful for MPI parallel programming
"""
from __future__ import print_function

from mpi4py import MPI

#=============================================================================
# I/O Utilities

def pprint(str="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(str+end, end=' ') 

