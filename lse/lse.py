import glob
import os
import random
import shutil
import time

# TODO consider the case where a worker process dies
class LSEWorker(object):

    # num_workers: total number of workers in the population
    # worker_id: identifier of this worker process
    # gen_ind: the number of individuals this worker process should generate for the initial population
    # population_dir: location for storing the population
    # population_lock: multiprocessing lock for concurrent file system access on systems that are not POSIX-compliant
    # remove_eliminated: Whether to remove individuals that are no longer part of the population from the file system
    def __init__(self, worker_id, si, population_lock, args):
        self.worker_id = worker_id
        self.si = si

        self.population_size = args.population_size
        self.generate_new = args.generate_new

        self.population_dir = args.population_dir
        if not os.path.exists(self.population_dir):
            os.makedirs(self.population_dir)
        self.population_lock = population_lock
        self.clean_population = args.clean_population

        self.ind_counter = 0
        self.log_dir = args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log = open(f"{self.log_dir}/log_{self.worker_id}.log", 'w')

        # If starting with an existing population, update the individual counter to prevent any future naming collisions
        population = glob.glob(f"{self.population_dir}/*_alive")
        for individual in population:
            ind_dir = os.path.split(individual)[1].split("_")
            if int(ind_dir[0]) == self.worker_id:
                if int(ind_dir[1]) + 1 > self.ind_counter:
                    self.ind_counter = int(ind_dir[1]) + 1
        self.generated_now = 0

    def run(self, ind_limit=None, time_limit=None):
        # Set a start time if the run is time-limited
        if time_limit is not None:
            time_start = time.time()

        # Generate some individuals to create an initial population
        for i in range(self.generate_new):
            # Generate an initial individual
            dna = self.si.generate_init()
            # Evaluate the new individual
            metrics = self.si.evaluate_ind(dna)

            # Create infrastructure for the new individual and store it
            individual = f"{self.population_dir}/{self.worker_id}_{self.ind_counter}_training"
            os.makedirs(individual)
            self.si.store_dna(individual, dna)
            # Also store metrics of the individual in separate files
            self.si.store_metrics(individual, metrics)

            # Rename the new individual's directory so it is added to the population
            # In some cases, the rename operation may fail for no apparent reason, so
            # keep trying until it is successful
            while True:
                try:
                    if os.path.exists(individual):
                        os.rename(
                            individual, f"{self.population_dir}/{self.worker_id}_{self.ind_counter}_alive")
                except:
                    continue
                break

            self.log.write(
                f"({time.time()}): Worker {self.worker_id} generated individual {self.ind_counter} with fitness {metrics['fitness']}\n")
            self.log.flush()

            # Increase the counter of generated individuals to ensure each individual has a
            # unique identifier
            self.ind_counter += 1
            self.generated_now += 1

        # Wait for the initial population to be completed
        while len(glob.glob(f"{self.population_dir}/*_alive")) < self.population_size:
            time.sleep(1.0)
        # Wait to ensure that all worker processes break the above loop
        time.sleep(2.0)

        # Continually evolve the population until a limit is reached
        # Note that when a limit is reached any model currently in training is finished
        while True:
            if ind_limit is not None and self.generated_now == ind_limit:
                break
            if time_limit is not None and time.time() - time_start >= time_limit:
                break

            self.evolve_population()

    def evolve_population(self):
        # A round of evolution may fail due to the parallel nature of the framework,
        # so keep trying until a round succeeds
        while True:
            # Get the individuals that are eligible for mutation
            population = glob.glob(f"{self.population_dir}/*_alive")
            # Check that the population isn't growing due to potential concurrency issues
            assert len(population) <= self.population_size
            # Sample two of those individuals uniformly at random without replacement
            # This is the only random operation in the evolutionary process, in case you're considering seeding for reproducibility
            # If you're already using NumPy and seeding it, it also has this operation available
            tournament = random.sample(population, k=2)
            # Retrieve the fitness of these two individuals
            metrics_one = self.si.read_metrics(tournament[0])
            metrics_two = self.si.read_metrics(tournament[1])
            # If either or both individuals are eliminated from the population between
            # sampling them and reading their fitness, abandon the round and start a new one
            if metrics_one is None or metrics_two is None:
                continue

            # Determine which of the two sampled individuals has better fitness
            if self.si.has_better_fitness(metrics_one["fitness"], metrics_two["fitness"]):
                best = tournament[0]
                worst = tournament[1]
            else:
                best = tournament[1]
                worst = tournament[0]

            # Read the model of the best individual as soon as possible, since it may
            # be removed from the population at any time by other workers
            dna = self.si.read_dna(best)
            # If either the best individual is eliminated from the population between sampling
            # and reading the model, abandon the round and start a new one
            if dna is None:
                continue

            # The following operation may cause the population to be corrupted if
            # it is not atomic. Since it is known not be atomic in some cases
            # (most notably in a Windows environment), the population can be locked
            # in such cases.
            if self.population_lock is not None:
                self.population_lock.acquire()

            # Rename the worst individual's directory. This prevents two workers from deleting
            # the same directory at the same time. Without this precaution, some workers will
            # think they deleted certain directories while in reality they didn't, causing the
            # population to grow over time.
            try:
                os.rename(worst, worst.replace("_alive", "_slated"))
            except:
                # Don't forget to release the lock in case renaming fails, if necessary
                if self.population_lock is not None:
                    self.population_lock.release()
                continue

            # Release the population lock if needed
            if self.population_lock is not None:
                self.population_lock.release()

            # Copy the model and mutate it, creating a new individual
            mutated_dna = self.si.mutate_ind(dna)
            # Evaluate the new individual
            mutated_metrics = self.si.evaluate_ind(mutated_dna)

            # Create infrastructure for the new individual and store it
            mutated_individual = f"{self.population_dir}/{self.worker_id}_{self.ind_counter}_training"
            os.makedirs(mutated_individual)
            self.si.store_dna(mutated_individual, mutated_dna)
            # Also store metrics of the individual in separate files
            self.si.store_metrics(mutated_individual, mutated_metrics)

            # The following operation may cause the population to be corrupted if
            # it is not atomic. Since it is known not be atomic in some cases
            # (most notably in a Windows environment), the population can be locked
            # in such cases.
            if self.population_lock is not None:
                self.population_lock.acquire()

            # Rename the new individual's directory so it is added to the population
            # In some cases, the rename operation may fail for no apparent reason, so
            # keep trying until it is successful
            while True:
                try:
                    if os.path.exists(mutated_individual):
                        os.rename(
                            mutated_individual, f"{self.population_dir}/{self.worker_id}_{self.ind_counter}_alive")
                except:
                    continue
                break

            # Release the population lock if needed
            if self.population_lock is not None:
                self.population_lock.release()

            self.log.write(
                f"({time.time()}): Worker {self.worker_id} generated individual {self.ind_counter} with fitness {mutated_metrics['fitness']}\n")
            self.log.flush()

            # Increase the counter of generated individuals to ensure each individual has a
            # unique identifier
            self.ind_counter += 1
            self.generated_now += 1

            # Remove the worst-performing of the two sampled individuals from the population
            # By replacing it with the new individual, the population size is kept constant
            # In some cases, the rename operation may fail for no apparent reason, so
            # keep trying until it is successful
            if self.clean_population:
                while True:
                    try:
                        if os.path.exists(worst.replace("_alive", "_slated")):
                            shutil.rmtree(worst.replace("_alive", "_slated"))
                    except:
                        continue
                    break
            else:
                while True:
                    try:
                        if os.path.exists(worst.replace("_alive", "_slated")):
                            os.rename(worst.replace("_alive", "_slated"),
                                      worst.replace("_alive", "_dead"))
                    except:
                        continue
                    break

            # This round is successful, so break from the retrying loop
            break
