import numpy as np
import random
import operator
import copy
import matplotlib.pyplot as plt
import file_maker as fm
import subprocess
# import pandas as pd


"""
chromosome[lr, init weight, optimizer, actFunc, layer, fitness]
lr = 0~1.0
init weight=  0:zeros    1:xavier    2:he
optimizer=    0:SGD      1:AdaGrad   2: Adam
activation=   0:sigmoid  1:relu
hidden layer= 1~10
fitness= accurate
"""


def dnn(gene):
    fm.fileMaker(gene)

    accuracy = subprocess.check_output("python3 text.py", shell=True)
    accuracy = str(accuracy)
    accuracy = accuracy[accuracy.find("Accuracy")+10:accuracy.find("genetic")]

    fitness = float(accuracy)
    return fitness


def getFitness(gene, popSize):
    for i in range(0, popSize):
        fitness = dnn(gene[i])
        gene[i][5] = fitness
    return gene


def select(sorted_chromosome, selectSize):
    selected_chromosome = []
    for i in range(0, selectSize):
        selected_chromosome.append(sorted_chromosome[i])
    return selected_chromosome


def mutate(selected_chromosome, mutateSize):
    mutated_chromosome = selected_chromosome
    for i in range(0, mutateSize):
        mutated_chromosome[i] = mutateGene(mutated_chromosome[i])
    return mutated_chromosome


def mutateGene(gene):
    mutated_gene = gene
    rand = random.randint(2, 3)
    choiceInd = random.sample([0, 1, 2, 3, 4], rand)
    for i in range(0, len(choiceInd)):
        if choiceInd[i] == 0:
            # mutate lr
            trans = random.randint(-20, 20)/100.0
            mutated_gene[0] = gene[0] + trans
        elif choiceInd[i] == 1:
            # mutate init weight
            mutated_gene[1] = random.choice(["zeros", "xavier", "he", "random"])
        elif choiceInd[i] == 2:
            # mutate optimizer
            mutated_gene[2] = random.choice(["SGD", "Adagrad", "Adam"])
        elif choiceInd[i] == 3:
            # mutate actFunc
            mutated_gene[3] = random.choice(["sigmoid", "relu", "tanh"])
        else:
            # mutate layer
            mutated_gene[4] = mutated_gene[4] + random.choice([-1, 1])
    return mutated_gene


def breed(selected_chromosome, popSize, breedSize):
    nextGeneration = selected_chromosome
    for i in range(0, breedSize):
        son = []
        randInd = random.sample(range(0, popSize-breedSize), 2)
        mom = selected_chromosome[randInd[0]]
        dad = selected_chromosome[randInd[1]]
        sep = random.randint(2, 5)
        for i in range(0, len(mom)):
            if i < sep:
                son.append(mom[i])
            else:
                son.append(dad[i])
        nextGeneration.append(son)
    return nextGeneration


# Init first generation
generation = 30
popSize = 100
mutateSize = 30
selectSize = 30
breedSize = 40
nextGeneration = []
for i in range(0, popSize):
    lr = random.randint(1, 100) / 100
    init_w = random.choice(["zeros", "xavier", "he", "random"])
    opt = random.choice(["SGD", "Adagrad", "Adam"])
    actF = random.choice(["sigmoid", "relu", "tanh"])
    layer = random.randint(2, 4)
    fitness = 0
    nextGeneration.append([lr, init_w, opt, actF, layer, fitness])

progress = []
# print(chromosome)
for i in range(0, generation):
    # print("Chromosome:", nextGeneration)
    getFitness(nextGeneration, popSize)
    sorted_chromosome = sorted(nextGeneration, key=operator.itemgetter(5), reverse=True)
    # print("after sort:", sorted_chromosome)
    selected_chromosome = select(sorted_chromosome, selectSize)
    # print("after select:", selected_chromosome)
    selected = copy.deepcopy(selected_chromosome)
    mutated_chromosome = mutate(selected_chromosome, mutateSize)
    # print("mutated_chromosome:", mutated_chromosome)
    nextGeneration = selected + mutated_chromosome
    # print("selected + mutate:", nextGeneration)
    nextGeneration = breed(nextGeneration, popSize, breedSize)
    # nextGeneration = sorted(nextGeneration, key=operator.itemgetter(5), reverse=True)
    # print("nextGeneration:", nextGeneration)
    progress.append(nextGeneration[0][5])
    # print("\n\n\n>>>>>Gen = ", i, "Max Accuracy:", nextGeneration[0], "\n\n\n")


# print("last Chromosome:", nextGeneration[0])
for i in range(popSize):
    print(i, "=", nextGeneration[i])
plt.plot(progress)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.show()
# print("next Generation:", nextGeneration)
