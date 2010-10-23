
=== Dependencies ===

These programs depend on mpi4py (>= Version 1.0)

The mpi4py documentation and installation instructions 
can be found at:

   http://mpi4py.scipy.org/

=== How to run on a single (multi-core) host ===

Run it with 

 mpirun -np 4 ./some-program

where the number after "-np " is the numer of parallel MPI 
processes to be started.


=== How to run on multiple hosts ===

If you want to run the program distributed over multiple hosts, 
you have to create a <hostfile> which looks like:

-- hostfile --
host1   slots=4
host2   slots=4
host3   slots=4
--------------

Where "slots=" specifies the number of parallel processes that should be
started on that host.

Run it with

  mpirun --hostfile <hostfile> ./some-program


=== Run on a cluster with the Torque Job schduling system ===

There are two possibilities:

a) Run interactively:

Request an interactive session and allocate a number of processors/nodes for 
your session:

 $ qsub -I X -l nodes=4:ppn=4

Where "-I" means you want to work interactively, "-X" requests grapical
(X-Window) I/O -- (you can run arbitrary programs that open windows).  The
option "-l " specifies the resources you want to allocate.  "-l nodes=4:ppn=4"
requests four compute nodes with each having four processor cores 
[ppn =^ ProcessorsPerNode].  So in total you allocate 16 CPU cores. 
[The scheduler is free to run your job on two nodes having 8 CPU cores each]

Once your interactive session is ready, you run 

 $ mpirun ./your-program
  
mpirun automatically knowns how many parallel processes have to be started and
where they have to be started.

b) Submit as non-interactive batch-job:

Use 

 $ qsub -l nodes=4:ppn=4 ./your-jobfile.job

to submit jour job-file. Similar to the interactive case, "-l" again is used  
to request resources from the scheduling system. The job file usually is a 
simple shell script which specifies the commands to be run once your job 
starts. In addition, the jobfile can contain "#PBS <something>" statements
which are used to specify additional options which could have been specified 
in the "qsub" commandline. Please see "man qsub" for details.

