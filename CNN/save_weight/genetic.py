import random
import operator
import copy
import file_writer as fm
import subprocess
import datetime
import os
import shutil

"""
    fitness = gene[0]
    lr = gene[1]
    initW = gene[2]
    optim = gene[3]
    actF = gene[4]
    kernel_size = gene[5]
    conv_layer = gene[6]
    fcn_layer = gene[7]
    dropout = gene[8]
    n_conv= gene[9]

"""


def cnn(gene, index, parent=None):
    fm.fileMaker(gene, index, parent)

    accuracy = subprocess.check_output("python3 created_cnn.py", shell=True)
    accuracy = str(accuracy)
    accuracy = accuracy[accuracy.find("Accuracy")+10:accuracy.find("genetic")]

    fitness = copy.deepcopy(float(accuracy))
    return fitness

def _getFitness(gene, popSize, selectSize):
    for i in range(selectSize, popSize):
        fitness = copy.deepcopy(cnn(gene[i], i))
        gene[i][0] = copy.deepcopy(fitness)

    return gene

def getFitness(gene, popSize, selectSize, parent):
    for i in range(selectSize, popSize):
        fitness = copy.deepcopy(cnn(gene[i], i, parent[i-selectSize]))
        gene[i][0] = copy.deepcopy(fitness)

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
    choiceInd = random.sample([1, 2, 3, 4, 5, 6, 7, 8], rand)
    for i in range(0, len(choiceInd)):
        if choiceInd[i] == 1:
            # mutate lr
            trans = random.randint(-3000, 3000)/10000.0
            mutated_gene[1] = abs(gene[0] + trans)
        elif choiceInd[i] == 2:
            # mutate init weight
            mutated_gene[2] = random.choice(["zeros", "he_uniform", "random_uniform"])
        elif choiceInd[i] == 3:
            # mutate optimizer
            mutated_gene[3] = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
        elif choiceInd[i] == 4:
            # mutate actFunc
            mutated_gene[4] = random.choice(["sigmoid", "relu", "tanh"])
        elif choiceInd[i] == 5:
            # mutate layer
            mutated_gene[5] = abs(mutated_gene[5] + random.choice([-2, 2]))
        elif choiceInd[i] == 6:
            # mutate conv_layer
            mutated_gene[6][0] = abs(mutated_gene[6][0] + random.choice([-1, 1]))  # conv_layer
            mutated_gene[6][1] = abs(mutated_gene[6][1] + random.choice([-1, 1]))  # n*conv in conv_layer
        elif choiceInd[i] == 7:
            # mutate fcnn_layer
            mutated_gene[7] = abs(mutated_gene[7] + random.choice([-1, 1]))
        elif choiceInd[i] == 8:
            d_rate = 0
            d_rate = random.uniform(0, 0.5)
            mutated_gene[8] = round(d_rate, 2)

    return mutated_gene

def breed(selected_chromosome, popSize, breedSize, plist):
    nextGeneration = selected_chromosome
    for i in range(0, breedSize):
        son = []
        randInd = random.sample(range(0, popSize-breedSize), 2)
        plist.append(randInd[0])
        mom = copy.deepcopy(selected_chromosome[randInd[0]])
        dad = copy.deepcopy(selected_chromosome[randInd[1]])
        sep = random.randint(2, 7)
        for j in range(0, len(mom)):
            if j < sep:
                son.append(mom[j])
            else:
                son.append(dad[j])
            son[0] = 0
        nextGeneration.append(son)

    return nextGeneration, plist

generation = 5
popSize = 5
mutateSize = 1
selectSize = 3
breedSize = 1
nextGeneration = []

for i in range(0, popSize):
    fitness = 0
    lr = random.randint(1, 10000) / 10000
    init_W = random.choice(["zeros", "he_uniform", "random_uniform"])
    opt = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
    actF = random.choice(["sigmoid", "relu", "tanh"])
    kernel_size = random.choice([1, 3, 5])
    conv_layer = random.choice([1, 2, 3])
    n_conv = random.randrange(1, 4)
    fcn_layer = random.choice([1, 2, 3])
    dropout = random.uniform(0, 0.5)
    nextGeneration.append([fitness, lr, init_W, opt, actF, kernel_size, [conv_layer, n_conv], fcn_layer, dropout, i])

now = datetime.datetime.now()
log = open("log.txt", 'w')
log.write("\n\n[first]\n")
for i in range(popSize):
    print("first = ", nextGeneration[i])
    log.write(str(nextGeneration[i])+"\n")
log.close()

progress = []

if not(os.path.isdir('saved')):
    os.makedirs(os.path.join('saved'))
if not(os.path.isdir('selected')):
    os.makedirs(os.path.join('selected'))

for i in range(0, generation):
    if i == 0:
        _getFitness(nextGeneration, popSize, 0)

    sorted_chromosome = copy.deepcopy(sorted(nextGeneration, key=operator.itemgetter(0), reverse=True))
    selected_chromosome = select(copy.deepcopy(sorted_chromosome), copy.deepcopy(selectSize))
    
    for file in os.listdir('selected'):
        os.remove('./selected/'file)
    for j in range(selectSize):
        shutil.move('./saved/model_' + str(selected_chromosome[j][9]) + '.h5', './selected/model_' + str(j) + '.h5')
        selected_chromosome[j][9] = j
        
    selected = copy.deepcopy(selected_chromosome)
    
    mutated_chromosome = mutate(copy.deepcopy(selected_chromosome), copy.deepcopy(mutateSize))
        
    plist = [0, 1, 2]
    nextGeneration = copy.deepcopy(selected) + copy.deepcopy(mutated_chromosome)
    nextGeneration, plist = breed(copy.deepcopy(nextGeneration), copy.deepcopy(popSize), copy.deepcopy(breedSize), copy.deepcopy(plist))

    for file in os.listdir('saved'):
        os.remove('./saved/', file)
    for j in range(popSize):
        if j < selectSize:
            shutil.move('./selected/model_' + str(j) + '.h5', './saved/')
        else:
            nextGeneration[j][9] = j
        f = open("./saved/chromosome_"+str(j)+".txt", 'w')
        f.write(str(nextGeneration[j][5])+'\n')
        f.write(str(nextGeneration[j][6][0])+'\n')
        f.write(str(nextGeneration[j][6][1])+'\n')
        f.write(str(nextGeneration[j][7])+'\n')
        f.close()

    getFitness(nextGeneration, popSize, selectSize, plist)
    progress.append(copy.deepcopy(nextGeneration[0][0]))
    log = open("log.txt", 'a')
    log.write("\ngeneration " + str(i) + " : " + str(nextGeneration[0][0]))
    log.close()

log = open("log.txt", 'a')
log.write("\n\n[last]\n")
print("OOOOOO Last Generation OOOOOO")
for i in range(popSize):
    print(i, "=", nextGeneration[i])
    log.write(str(nextGeneration[i])+"\n")
log.close()
print(" progress =", progress)
