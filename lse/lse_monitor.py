import glob
import os
import sys
import time

from lse_cnn import CNNInterface

# Get the best fitness from the population in the provided dir.
# The provided interface is used to compare fitnesses.
def get_best_fitness(population_dir, si):
    population = glob.glob(f"{population_dir}/*_alive")
    best_fitness = None
    for individual in population:
        metrics = si.read_metrics(individual)
        if metrics is None:
            continue
        if best_fitness is None or si.has_better_fitness(metrics['fitness'], best_fitness):
            best_fitness = metrics['fitness']
    return best_fitness


def get_pop_fitness(population_dir, si):
    population = glob.glob(f"{population_dir}/*_alive")
    pop2 = glob.glob(f"{population_dir}/*_slated")
    fitness = []
    for individual in population + pop2:
        metrics = si.read_metrics(individual)
        if metrics is None:
            continue
        fitness.append(metrics['fitness'])
    return fitness

# Requires one command-line argument: the population directory
if __name__ == '__main__':
    while True:
        print(get_best_fitness("population", CNNInterface(sys.argv[1])))
        time.sleep(0.5)
