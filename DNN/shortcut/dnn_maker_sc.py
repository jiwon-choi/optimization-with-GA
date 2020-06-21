import numpy as np
import random
import operator
import copy
import maker as fm
import subprocess
# import pandas as pd




def dnn(gene):
    fm.fileMaker(gene)

    accuracy = subprocess.check_output("python3 created_dnn.py", shell=True)
    accuracy = str(accuracy)
    accuracy = accuracy[accuracy.find("Accuracy")+10:accuracy.find("normal_dnn")]

    fitness = float(accuracy)
    return fitness


gene = [0.001, "zeros", "Adam", "relu", 100, 0]

gene[5] = dnn(gene)

print("--Acc--")
print(gene[5])
