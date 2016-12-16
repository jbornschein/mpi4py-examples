""" A different implementation of task-pull with less communication and full
use of resources (mainly-idle parent shares with worker). Sentinels are used in
place of tags. Start parent with 'python <filename.py>' rather than mpirun;
parent will then spawn specified number of workers. Work is randomized to
demonstrate dynamic allocation. Worker logs are collectively passed back to
parent at the end in place of results. Comments and output are both
deliberately excessive for instructional purposes. """
from __future__ import print_function
from __future__ import division

from mpi4py import MPI
import random
import time
import sys

n_workers = 10
n_tasks = 50
start_worker = 'worker'
usage = 'Program should be started without argument'

# Parent
if len(sys.argv) == 1:

    # Start clock
    start = MPI.Wtime()

    # Random 1-9s tasks
    task_list = [random.randint(1, 9) for task in range(n_tasks)]
    total_time = sum(task_list)

    # Append stop sentinel for each worker
    msg_list = task_list + ([StopIteration] * n_workers)

    # Spawn workers
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[sys.argv[0], start_worker],
        maxprocs=n_workers)

    # Reply to whoever asks until done
    status = MPI.Status()
    for position, msg in enumerate(msg_list):
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())

        # Simple (loop position) progress bar
        percent = ((position + 1) * 100) // (n_tasks + n_workers)
        sys.stdout.write(
            '\rProgress: [%-50s] %3i%% ' %
            ('=' * (percent // 2), percent))
        sys.stdout.flush()

    # Gather reports from workers
    reports = comm.gather(root=MPI.ROOT)

    # Print summary
    workers = 0; tasks = 0; time = 0
    print('\n\n  Worker   Tasks    Time')
    print('-' * 26)
    for worker, report in enumerate(reports):
        print('%8i%8i%8i' % (worker, len(report), sum(report)))
        workers += 1; tasks += len(report); time += sum(report)
    print('-' * 26)
    print('%8i%8i%8i' % (workers, tasks, time))

    # Check all in order
    assert workers == n_workers, 'Missing workers'
    assert tasks == n_tasks, 'Lost tasks'
    assert time == total_time, 'Output != assigned input'

    # Final statistics
    finish = MPI.Wtime() - start
    efficiency = (total_time * 100.) / (finish * n_workers)
    print('\nProcessed in %.2f secs' % finish)
    print('%.2f%% efficient' % efficiency)

    # Shutdown
    comm.Disconnect()

# Worker
elif sys.argv[1] == start_worker:

    # Connect to parent
    try:
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()
    except:
        raise ValueError('Could not connect to parent - ' + usage)

    # Ask for work until stop sentinel
    log = []
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):
        log.append(task)

        # Do work (or not!)
        time.sleep(task)

    # Collective report to parent
    comm.gather(sendobj=log, root=0)

    # Shutdown
    comm.Disconnect()

# Catch
else:
    raise ValueError(usage)
