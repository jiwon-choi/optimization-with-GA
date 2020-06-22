import random
import operator
import copy
import cnn_maker_v_ as fm
import subprocess
import datetime


def cnn(gene):
    fm.fileMaker(gene)
    print("aa")

    accuracy = subprocess.check_output("python3 created_cnn.py", shell=True)
    accuracy = str(accuracy)
    accuracy = accuracy[accuracy.find("Accuracy")+10:accuracy.find("genetic")]

    fitness = copy.deepcopy(float(accuracy))
    return fitness


gene = [0, 0.001, "he_uniform", "Adam", "relu", 3, [3, 2], 2, 0.25]
fit = cnn(gene)

print(fit)
