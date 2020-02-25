import numpy as np
import random
import operator
import copy
import optimize_structure as fm
import subprocess
import time
# import pandas as pd


"""
    lr = gene[0]
    initW = gene[1]
    optim = gene[2]
    actF = gene[3]
    kernel_size = gene[4]
    conv_layer = gene[5]
    fcn_layer = gene[6]
"""


def cnn(gene):
    fm.fileMaker(gene)
    
    accuracy = subprocess.check_output("python3 created_cnn.py", shell=True)
    accuracy = str(accuracy)
    accuracy = accuracy[accuracy.find("Accuracy")+10:accuracy.find("genetic")]
    
    fitness = copy.deepcopy(float(accuracy))
    return fitness


def getFitness(gene, popSize, selectSize):
    for i in range(selectSize, popSize):
        fitness = copy.deepcopy(cnn(gene[i]))
        gene[i][7] = copy.deepcopy(fitness)
        
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
    rand = random.randint(4, 5)
    choiceInd = random.sample([0, 1, 2, 3, 4, 5, 6], rand)
    for i in range(0, len(choiceInd)):
        if choiceInd[i] == 0:
            # mutate lr
            trans = random.randint(-3000, 3000)/10000.0
            mutated_gene[0] = abs(gene[0] + trans)
        elif choiceInd[i] == 1:
            # mutate init weight
            mutated_gene[1] = random.choice(["zeros", "he_uniform", "random_uniform"])
        elif choiceInd[i] == 2:
            # mutate optimizer
            mutated_gene[2] = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
        elif choiceInd[i] == 3:
            # mutate actFunc
            mutated_gene[3] = random.choice(["sigmoid", "relu", "tanh"])
        elif choiceInd[i] == 4:
            # mutate layer
            mutated_gene[4] = abs(mutated_gene[4] + random.choice([-2, 2]))
        elif choiceInd[i] == 5:
            #mutate conv_layer
            mutated_gene[5] = abs(mutated_gene[5] + random.choice([-1, 1]))
        elif choiceInd[i] == 6:
            #mutate conv_layer
            mutated_gene[6] = abs(mutated_gene[6] + random.choice([-1, 1]))
    
    return mutated_gene


def breed(selected_chromosome, popSize, breedSize):
    nextGeneration = selected_chromosome
    for i in range(0, breedSize):
        son = []
        randInd = random.sample(range(0, popSize-breedSize), 2)
        mom = selected_chromosome[randInd[0]]
        dad = selected_chromosome[randInd[1]]
        sep = random.randint(2, 6)
        for i in range(0, len(mom)):
            if i < sep:
                son.append(mom[i])
            else:
                son.append(dad[i])
        nextGeneration.append(son)
    son[7] = 0
    # print("son=",son)
    return nextGeneration

# main


generation = 20
popSize = 30
mutateSize = 9
selectSize = 9
breedSize = 12
nextGeneration = []
for i in range(0, popSize):
    lr = random.randint(1, 10000) / 10000
    init_w = random.choice(["zeros", "he_uniform", "random_uniform"])
    opt = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
    actF = random.choice(["sigmoid", "relu", "tanh"])
    kernel_size = random.choice([1, 3, 5])
    conv_layer = random.choice([1, 2, 3])
    fcn_layer = random.choice([1, 2, 3])
    fitness = 0
    nextGeneration.append([lr, init_w, opt, actF, kernel_size, conv_layer, fcn_layer, fitness])


now = time.localtime()
# strnow = "log_"+str(now.tm_year)+"-"+str(now.tm_mon)+"-"+str(now.tm_mday)+"_"+str(now.tm_hour)+"-"+str(now.tm_min)
log = open("log200225.txt", 'a')
log.write("\n\n[first]\n")
for i in range(popSize):
    print("first =", nextGeneration[i])
    log.write(str(nextGeneration[i])+"\n")
log.close()

progress = []
for i in range(0, generation):
    if i == 0:
        getFitness(nextGeneration, popSize, 0)
    
    sorted_chromosome = copy.deepcopy(sorted(nextGeneration, key=operator.itemgetter(7), reverse=True))
    selected_chromosome = copy.deepcopy(select(sorted_chromosome, selectSize))
    selected = copy.deepcopy(selected_chromosome)
    mutated_chromosome = copy.deepcopy(mutate(selected_chromosome, mutateSize))
    nextGeneration = copy.deepcopy(selected) + copy.deepcopy(mutated_chromosome)
    nextGeneration = copy.deepcopy(breed(nextGeneration, popSize, breedSize))
    getFitness(nextGeneration, popSize, selectSize)
    progress.append(copy.deepcopy(nextGeneration[0][7]))
    log = open("log200225.txt", 'a')
    log.write("\ngeneration " + str(i)+ " : " + str(nextGeneration[0][7]))
    log.close()

log = open("log200225.txt", 'a')
log.write("\n\n[last]\n")
print("OOOOOO Last Generation OOOOOO")
for i in range(popSize):
    print(i, "=", nextGeneration[i])
    log.write(str(nextGeneration[i])+"\n")
log.close()
print(" progress =", progress)
