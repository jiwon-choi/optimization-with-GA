import random
import operator
import copy
import cnn_maker_v_ as fm
import subprocess
import datetime


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
        gene[i][0] = copy.deepcopy(fitness)

    return gene


def select(sorted_chromosome, selectSize):
    selected_chromosome = []
    for i in range(0, selectSize):
        selected_chromosome.append(sorted_chromosome[i])

    return selected_chromosome


def mutateGene(gene):
    mutated_gene = copy.deepcopy(gene)
    rand = random.randint(4, 5)
    choiceInd = random.sample([1, 2, 3, 4, 5, 6, 7, 8], rand)
    for i in range(0, len(choiceInd)):
        if choiceInd[i] == 1:
            # mutate lr
            trans = random.randint(-3000, 3000)/10000.0
            mutated_gene[0] = abs(gene[0] + trans)
        elif choiceInd[i] == 2:
            # mutate init weight
            mutated_gene[1] = random.choice(["zeros", "he_uniform", "random_uniform"])
        elif choiceInd[i] == 3:
            # mutate optimizer
            mutated_gene[2] = random.choice(["SGD", "Adagrad", "Adam", "Adadelta"])
        elif choiceInd[i] == 4:
            # mutate actFunc
            mutated_gene[3] = random.choice(["sigmoid", "relu", "tanh"])
        elif choiceInd[i] == 5:
            # mutate layer
            mutated_gene[4] = abs(mutated_gene[4] + random.choice([-2, 2]))
        elif choiceInd[i] == 6:
            # mutate conv_layer
            mutated_gene[5][0] = abs(mutated_gene[5][0] + random.choice([-1, 1]))  # conv_layer
            mutated_gene[5][1] = abs(mutated_gene[5][1] + random.choice([-1, 1]))  # n*conv in conv_layer
        elif choiceInd[i] == 7:
            # mutate fcnn_layer
            mutated_gene[6] = abs(mutated_gene[6] + random.choice([-1, 1]))
        elif choiceInd[i] == 8:
            d_rate = 0
            d_rate = random.uniform(0, 0.5)
            mutated_gene[7] = round(d_rate, 2)

    return mutated_gene

def breed(selected_chromosome, popSize, breedSize):
    nextGeneration = selected_chromosome
    for i in range(0, breedSize):
        son = []
        randInd = random.sample(range(0, popSize-breedSize), 2)
        mom = copy.deepcopy(selected_chromosome[randInd[0]])
        dad = copy.deepcopy(selected_chromosome[randInd[1]])
        sep = random.randInt(2, 7)
        for j in range(0, len(mom)):
            if j < sep:
                son.append(mom[j])
            else:
                son.append(dad[j])
            son[0] = 0
        nextGeneration.append(son)

    return nextGeneration

generation = 5
popSize = 10
mutateSize = 3
selectSize = 3
breedSize = 4
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
    nextGeneration.append([fitness, lr, init_w, opt, actF, kernel_size, [conv_layer, n_conv], fcn_layer, dropout])

now = datetime.datetime.now()
log = open("log.txt", 'a')
log.write("\n\n[first]\n")
for i in range(popSize):
    print("first = ", nextGeneration[i])
    log.write(str(nextGeneration[i])+"\n")
log.close()

progress = []
for i in range(0, generation):
    if i == 0:
        getFitness(nextGeneration, popSize, 0)

    sorted_chromosome = copy.deepcopy(sorted(nextGeneration, key=operator.itemgetter(0), reverse=True))
    selected_chromosome = select(copy.deepcopy(sorted_chromosome), copy.deepcopy(selectSize))
    selected = copy.deepcopy(selected_chromosome)
    mutated_chromosome = mutate(copy.deepcopy(selected_chromosome), copy.deepcopy(mutateSize))
    nextGeneration = copy.deepcopy(selected) + copy.deepcopy(mutated_chromosome)
    nextGeneration = breed(copy.deepcopy(nextGeneration), copy.deepcopy(popSize), copy.deepcopy(breedSize))
    getFitness(nextGeneration, popSize, selectSize)
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
