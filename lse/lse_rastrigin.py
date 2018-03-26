import math
import pickle
import random
import time

# Example of an interface for the evolutionary process. This particular
# interface can be used to optimize the solution to the Rastrigin function.
# This is a minimization problem, where the DNA size is the dimensionality
# of of the solution space.
class RastriginInterface(object):

    # Use the constructors to setup the setting.
    # Note that these values will be accessed will generally be accessed by
    # multiple processes concurrently, so limit changing values in self to
    # objects that are process-safe.
    def __init__(self, dna_size):
        self.dna_size = dna_size

    # Generate and return the  DNA for an individual for the initial population
    def generate_init(self):
        dna = [random.uniform(-5.12, 5.12) for i in range(self.dna_size)]

        return dna

    # Read and return DNA of an individual, or None if the DNA no longer exists
    def read_dna(self, individual):
        try:
            with open(f"{individual}/dna.pkl", 'rb') as dna_file:
                return pickle.load(dna_file)
        except:
            return None

    # Store the DNA of an individual
    def store_dna(self, individual, dna):
        with open(f"{individual}/dna.pkl", 'wb') as dna_file:
            pickle.dump(dna, dna_file, pickle.HIGHEST_PROTOCOL)

    # Read and return the metrics of an individual, or None if the metrics no longer exist
    def read_metrics(self, individual):
        try:
            with open(f"{individual}/metrics.pkl", 'rb') as metrics_file:
                return pickle.load(metrics_file)
        except:
            return None

    # Store the metrics of an individual. The metrics must be passed in the form of a
    # dictionary with at least the key "fitness" set
    def store_metrics(self, individual, metrics):
        with open(f"{individual}/metrics.pkl", 'wb') as metrics_file:
            pickle.dump(metrics, metrics_file, pickle.HIGHEST_PROTOCOL)

    # Evaluate the fitness of the individual represented by the DNA. Return a dictionary
    # with at least the key "fitness" set
    def evaluate_ind(self, dna):
        start_time = time.time()
        fitness = 10 * \
            len(dna) + sum([x**2 - 10 * math.cos(2 * math.pi * x)
                            for x in dna])
        evaluation_time = time.time() - start_time

        metrics = {}
        metrics["fitness"] = fitness
        metrics["evaluation_time"] = evaluation_time

        # Remove this line if you want to generate A LOT of individuals very fast
        time.sleep(random.uniform(0.1, 0.5))

        return metrics

    # Check whether the first fitness is better than the second fitness
    def has_better_fitness(self, fitness_one, fitness_two):
        return fitness_one < fitness_two

    # Mutate the provided DNA a single time
    def mutate_ind(self, dna):
        # Compose the list of available mutations
        mutations = [self.random_mutation]
        success = False

        # Keep trying to apply a mutation until one is successful
        while not success:
            mutated_dna, success = random.choice(mutations)(dna)

        return mutated_dna

    # Randomly set one of the DNA values
    def random_mutation(self, dna):
        mutated_dna = dna[:]
        mutated_dna[random.randint(0, len(dna) - 1)
                    ] = random.uniform(-5.12, 5.12)

        return mutated_dna, True
