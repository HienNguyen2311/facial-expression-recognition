import random
import numpy as np
from deap import tools
import dill as pickle
import os

# Define protected division
def protected_div(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=y!=0)

def protected_log(x):
    return np.log(np.abs(x) + 1e-10)

def protected_sqrt(x):
    return np.sqrt(np.abs(x))

# Define vector operations
def vector_add(a, b):
    return np.add(a, b)

def vector_sub(a, b):
    return np.subtract(a, b)

def vector_mul(a, b):
    return np.multiply(a, b)

def vector_div(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b!=0)

def vector_sin(a):
    return np.sin(a)

def vector_cos(a):
    return np.cos(a)

def vector_exp(a):
    return np.exp(a)

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb)
    # Apply crossover and mutation on the offspring
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, FREQ=None, chkpoints_dir=None, start_gen=1):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)
    for gen in range(start_gen, ngen + 1):
        #Select the next generation individuals by elitism
        elitismNum=int(elitpb * len(population))
        population_for_eli=[toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        offspring = toolbox.select(population, len(population)-elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        for i in offspring:
            ind = 0
            while ind<len(hof_store):
                if i == hof_store[ind]:
                    i.fitness.values = hof_store[ind].fitness.values
                    ind = len(hof_store)
                else:
                    ind+=1

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring[0:0]=offspringE
            
        # Update the hall of fame with the generated
        if halloffame is not None:
            halloffame.update(offspring)
        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

        # Save checkpoint every FREQ generations
        if gen % FREQ == 0:
            cp_filename = os.path.join(chkpoints_dir, f"IDGP_checkpoint_gen_{gen}.pkl")
            with open(cp_filename, "wb") as cp_file:
                pickle.dump({
                    "population": population,
                    "generation": gen,
                    "halloffame": halloffame,
                    "logbook": logbook,
                    "rndstate": random.getstate(),
                }, cp_file)
            print(f"Checkpoint saved to {cp_filename} at generation {gen}")

    return population, logbook
