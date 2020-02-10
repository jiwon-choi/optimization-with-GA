import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis**2)+(yDis**2))
        return distance

    def __repr__(self):
        return "("+str(self.x) + "," + str(self.y) + ")"

# Fitness class
# fitness = 1 / (route distance)
# ㅇㅇ


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i+1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1/float(self.routeDistance())
        return self.fitness


def initialPopulation(pop_size, cityList):
    population = []

    for i in range(0, pop_size):
        population.append(createRoute(cityList))
    return population


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(population, popRanked, elite_size, prob, cityList):
    selected_gene = []
    e = elite_size
    for i in range(0, elite_size):
        ind = popRanked[i][0]
        selected_gene.append(population[ind])

    for i in range(0, len(population)-elite_size):

        for j in range(0, len(population)):
            pick = random.random()
            if(pick <= 2/3 * ((1/3)**(j-1))):
                # print("picked!")
                e = e+1
                ind = popRanked[j][0]
                selected_gene.append(population[ind])
                break

    for i in range(0, len(population)-e):
        selected_gene.append(createRoute(cityList))

    return selected_gene


def mutate(population, mutationRate):
    m_population = []
    # print("====original====\n",population)
    # print("====after mutate====")
    for i in range(0, len(population)):
        mr = random.random()
        # print(mr)
        if(mr <= mutationRate):
            # print("(mutate in %d)"%i)
            temp = population[i][1:len(population)-2]
            random.shuffle(temp)
            population[i][1:len(population)-2] = temp
            m_population.append(population[i])
        else:
            m_population.append(population[i])

    return m_population


def oneCycle(currentGen, elite_size, mutate_rate, prob):
    mutated_pop = mutate(currentGen, mutate_rate)
    popRanked = rankRoutes(mutated_pop)
    nextGeneration = selection(currentGen, popRanked, elite_size, prob, cityList)
    # nextGeneration = mutate(selectionResults, mutate_rate)
    return nextGeneration


def geneticAlgorithm(population, pop_size, elite_size, mutate_rate, generations, prob):
    pop = initialPopulation(pop_size, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    print("Initial distance:" + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = oneCycle(pop, elite_size, mutate_rate, prob)
        progress.append(1 / rankRoutes(pop)[0][1])
    print("Final distance:" + str(1 / rankRoutes(pop)[0][1]))
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# main


city_num = 25
pop_size = 100
mutate_rate = 0.01
elite_size = 90
generations = 500
prob = 1

cityList = []
for i in range(0, city_num):
    cityList.append(City(x=int(random.random()*200), y=int(random.random()*200)))

geneticAlgorithm(cityList, pop_size, elite_size, mutate_rate, generations, prob)

'''
population = initialPopulation(pop_size, cityList)
mutated_pop = mutate(population, mutate_rate)
# print(mutated_pop)
rank = rankRoutes(mutated_pop)
# print(rank)

print(selection(population, rank, elite_size, prob))
# print(mutated_pop)
'''
