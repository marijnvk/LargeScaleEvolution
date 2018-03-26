import argparse
import glob
from multiprocessing import Lock, Process
import os
import shutil

from lse import LSEWorker
from lse_cnn import CNNInterface
#from lse_rastrigin import RastriginInterface


# Setting up the LSEWorker here instead of before the process starts prevents some pickling issues
def __lse_starter(i, si, population_lock, args):
    lse = LSEWorker(i, si, population_lock, args)
    lse.run(ind_limit=args.generate_limit, time_limit=args.time_limit)


def arg_bool_to_bool(arg_bool):
    if arg_bool in ('True', 'true', 'T', 't', '1', 'Y', 'y'):
        return True
    elif arg_bool in ('False', 'false', 'F', 'f', '0', 'N', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Some notes on the command-line arguments:
#
# num_workers     : If you call this from another python script, an option is to use os.cpu_count() or a mutliple of it for this argument.
# first_worker_id : Use this if you want to start additional processes in a run that is currently active, or you can use it if you want to use new IDs when resuming a run that was terminated earlier.
# population_size : Workers will not start the evolutionary process until the population reaches this size, so make sure enough initial individuals are generated.
# generate_new    : The number of new initial individuals to generate in total. This work is distributed equally among workers.
# reset_population: If false, individuals in population_dir are deleted if they are in a training state and reinstated as alive if they are slated. If true, all individuals are deleted.
# time_limit      : Workers will finish any training in progress before terminating when the time limit is reached. The limit it tracked per worker and starts when the worker starts generating initial individuals.
# generate_limit  : The maximum number of individuals each worker may generate. This includes any initial individuals. The more individual training time and/or evaluation time varies, the more time difference there can be between workers finishing when only using this limit (and not an explicit time limit).
# population_dir  : If this is an existing directory, ensure that it and its contents are not read-only.
# log_dir         : This directory stores the logs of the worker processes. The logs of individuals may be stored elsewhere, depending on the interface that is used. If logs for individuals are stored in the population directory, they may be lost if individuals are deleted.
# software_locks  : Software locking of the population will automatically be performed if the system is not POSIX-compliant. Use this argument to force the (non-)usage of software locks.
# clean_population: If true, individuals are fully removed from the file system when eliminated from the population. If false, they are only renamed to indicate their elimination. Note that this can cause the number of directories in the population directory to grow very large very fast, which may cause issues with the maximum number of sub-directories the file system allows.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", "-w",
                        help="Number of worker processes to run concurrently", default=1, type=int)
    parser.add_argument("--first_worker_id", "-f",
                        help="Which number to start numbering worker processes from", default=0, type=int)
    parser.add_argument("--generate_new", "-g",
                        help="Number of initial individuals to add to the population", default=4, type=int)

    parser.add_argument("--population_size", "-p",
                        help="Number of individuals in the population", default=4, type=int)
    parser.add_argument("--reset_population", "-r",
                        help="Whether to remove any existing individuals", default=False, type=arg_bool_to_bool)
    parser.add_argument("--population_dir", "-d",
                        help="Directory for storing all individuals", default="population", type=str)
    parser.add_argument("--clean_population", "-c",
                        help="Whether to fully delete individuals that are removed from the population", default=False, type=arg_bool_to_bool)

    parser.add_argument(
        "--time_limit", "-t", help="Running time limit in seconds", default=None, type=float)
    parser.add_argument("--generate_limit", "-n",
                        help="Maximum number of individuals to generate per worker", default=None, type=int)

    parser.add_argument(
        "--log_dir", "-l", help="Directory for storing worker logs", default="logs", type=str)
    parser.add_argument("--software_locks", "-s",
                        help="Force (non-)usage of software locks", default=None, type=arg_bool_to_bool)

    args = parser.parse_args()

    if args.reset_population == 1:
        shutil.rmtree(args.population_dir)
    else:
        population = glob.glob(os.path.join(args.population_dir, "*"))
        for individual in population:
            if individual.endswith("_slated"):
                os.rename(individual, individual.replace("_slated", "_alive"))
            elif individual.endswith("_training"):
                shutil.rmtree(individual)

    # Set the desired context here
    #si = RastriginInterface(10)
    si = CNNInterface()

    population_lock = None
    if args.software_locks is None:
        if os.name != "posix":
            population_lock = Lock()
        remove_eliminated = True
    elif args.software_locks:
        population_lock = Lock()

    # TODO make sure all args are checked
    for i in range(args.first_worker_id, args.first_worker_id + args.num_workers):
        p = Process(target=__lse_starter, args=(i, si, population_lock, args))
        p.start()
