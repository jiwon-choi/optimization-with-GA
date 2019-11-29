

# 1.하나의 도시는 반드시 한번만 거친다
# 2.처음의 도시로 돌아와야하며, 도시를 거칠때마다 총 이동거리를 구해준다

# Gene: (x,y)로 구성된 각 도시
# Chromosome: 위 조건을 만족하는 하나의 이동경로
# Population: 가능한 경로의 집단(i.e., collection of chromosome)
# Parents: 하나의 새로운 경로를 만들기위해 엮인 두 경로
# Mating pool: Parents의 집단?
# Fitness: 경로가 짧은 정도
# Mutation: (블로그) 경로에서 랜덤하게 두개의 도시를 바꿔줌
# Elitism: 우월주의(?) 다음세대로 우수한 개체들만 보내는 방법

# GA의 대략적 구조
# 1. Create the Population
# 2. Determine Fitness
# 3. Selecting Mating pool
# 4. Breed(= 번식?복제?)
# 5. Mutate
# 6. Repeat

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

# City class를 만들어줌
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
# 따라서 Fitness값이 클수록 좋다
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0,len(self.route)):
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
        if self.fitness  ==0:
            self.fitness = 1/ float(self.routeDistance())
        return self.fitness

# Initializing Population
# first generation을 만들어준다

def createRoute(cityList):
    route = random.sample(cityList, len(cityList)) # 중복,정렬 없이 랜덤하게 뽑아냄
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse=True)
    '''
    fitnessResults.items()을 하면 [(1,fitness1),(2,fitness2)]처럼 값을 반환함
    operator.itemgetter(1)은 1번째 인덱스값으로 오름차순하여 정렬 한다는 뜻
    거기에 reverse = True 를 했으니, Fitness값이 큰 population부터 정렬 해준다
    '''
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    #df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    df['cum_perc'] = 100*df.Fitness/df.Fitness.sum()
    '''
    index   Fitness     cum_sum     cum_perc
    0           1           1          5
    1           3           4          20
    2           6           10         50
    3           10          20         100
    위는 예시 이고 실제로 popRanked는 Fitness값을 내림차순으로 한 리스트이므로 아래와 같은 결과가 나온다


       index   Fitness     cum_sum     cum_perc
        0          10          10         50
        1           6          16         80
        2           3          19         96
        3           1          20         100
    이런식으로 결과값이 나오는데 cum_perc를 구할때 100*df.Fitness/df.Fitness.sum()이 더 적절하지 않을까 싶음
    '''

    '''
    여기부터 주석 다시 달아줘야함
    '''

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:   # iat[i,3] 에사 3은 어디서 나타난거??
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1)) # 여긴 왜 parent1?

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)  # 이 두줄은 무슨의미? 아 분리점을 정해주는듯

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatePop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatePop.append(mutatedInd)
    return mutatePop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance:" + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance:" + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    print("Initial distance:" + str(1 / rankRoutes(pop)[0][1]))


    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    print("Final distance:" + str(1 / rankRoutes(pop)[0][1]))
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

# main
cityList = []

for i in range(0,25):
    cityList.append(City(x=int(random.random()*200), y=int(random.random()*200)))
#geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
