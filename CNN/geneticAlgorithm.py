import numpy as np
import random
import operator
import copy
import createCNN as fm
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

    accuracy = subprocess.check_output("python3 created_cnn.py", shell=True)
    accuracy = str(accuracy)
    accuracy = accuracy[accuracy.find("Accuracy")+10:accuracy.find("genetic")]

    fitness = float(accuracy)
    return fitness


def getFitness(gene, popSize, selectSize):
    for i in range(selectSize, popSize):
        fitness = copy.deepcopy(dnn(gene[i]))
        gene[i][5] = copy.deepcopy(fitness)
    return gene


def select(sorted_chromosome, selectSize):
    selected_chromosome = []
    for i in range(0, selectSize):
        selected_chromosome.append(sorted_chromosome[i])
    return selected_chromosome


def mutate(selected_chromosome, mutateSize):
    before_chromosome = copy.deepcopy(selected_chromosome)
    mutated_chromosome = []
    for i in range(0, mutateSize):
        mutated_chromosome.append(mutateGene(before_chromosome[i]))
    return mutated_chromosome


def mutateGene(gene):
    mutated_gene = copy.deepcopy(gene)
    rand = random.randint(2, 3)
    choiceInd = random.sample([0, 1, 2, 3, 4], rand)
    for i in range(0, len(choiceInd)):
        if choiceInd[i] == 0:
            # mutate lr
            trans = random.randint(-20, 20)/1000.0
            mutated_gene[0] = gene[0] + trans
        elif choiceInd[i] == 1:
            # mutate init weight
            mutated_gene[1] = random.choice(["zeros", "xavier", "he", "random"])
        elif choiceInd[i] == 2:
            # mutate optimizer
            mutated_gene[2] = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
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
    son[5] = 0
    # print("son=",son)
    return nextGeneration


# Init first generation
generation = 2
popSize = 10
mutateSize = 3
selectSize = 3
breedSize = 4
nextGeneration = []
for i in range(0, popSize):
    lr = random.randint(1, 1000) / 1000
    init_w = random.choice(["zeros", "xavier", "he", "random"])
    opt = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
    actF = random.choice(["sigmoid", "relu", "tanh"])
    layer = random.randint(2, 4)
    fitness = 0
    nextGeneration.append([lr, init_w, opt, actF, layer, fitness])

for i in range(popSize):
    print("first =", nextGeneration[i])
progress = []
# print(chromosome)
for i in range(0, generation):
    # print("Chromosome:", nextGeneration)
    if i != 0:
        getFitness(nextGeneration, popSize, selectSize)
    else:
        getFitness(nextGeneration, popSize, 0)
    sorted_chromosome = sorted(nextGeneration, key=operator.itemgetter(5), reverse=True)
    '''
    for i in range(popSize):
        print(i, "=", sorted_chromosome[i])
    '''
    selected_chromosome = select(sorted_chromosome, selectSize)
    # print("after select:", selected_chromosome)
    selected = copy.deepcopy(selected_chromosome)
    mutated_chromosome = mutate(selected_chromosome, mutateSize)
    # print("mutated_chromosome:", mutated_chromosome)
    nextGeneration =  selected + copy.deepcopy(mutated_chromosome)
    # print("selected + mutate:", nextGeneration)
    nextGeneration = breed(nextGeneration, popSize, breedSize)
    # nextGeneration = sorted(nextGeneration, key=operator.itemgetter(5), reverse=True)
    # print("nextGeneration:", nextGeneration)
    progress.append(nextGeneration[0][5])

    # print("\n\n\n>>>>>Gen = ", i, "Max Accuracy:", nextGeneration[0], "\n\n\n")


# print("last Chromosome:", nextGeneration[0])
print("—Last Generation—")
for i in range(popSize):
    print(i, "=", nextGeneration[i])
print(" progress =", progress)
# plt.plot(progress)
# plt.ylabel('Fitness')
# plt.xlabel('Generation')
# plt.show()
# print("next Generation:", nextGeneration)
