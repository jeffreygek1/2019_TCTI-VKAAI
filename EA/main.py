from functools import reduce
from random import shuffle, random, randint


class EA:
    # init population
    def __init__(self, n):
        self.population = self.genPopulation(n)

    # Generate random individual of type list of card numbers
    def genIndividual(self):
        cards = [i for i in range(1, 11)]
        shuffle(cards)
        return cards

    # Generate population of individuals of type list of card numbers
    def genPopulation(self, n):
        return [self.genIndividual() for i in range(n)]

    # Test individual of its fitness
    def getFitness(self, individual, n=5):
        pile1 = individual[:n]
        pile2 = individual[n:]

        result1 = sum(pile1)
        result2 = reduce(lambda x, y: x * y, pile2)

        fit1 = abs(36 - result1)
        fit2 = abs(360 - result2)

        return fit1 + fit2

    # Generate children based in parents
    def makeChild(self, mother, father):
        gnomeSize = len(mother)
        gnomePosition = []
        while int(gnomeSize / 2) > len(gnomePosition):
            r = randint(0, 9)
            if r not in gnomePosition:
                gnomePosition.append(r)

        child1 = [0 for i in range(gnomeSize)]
        child2 = [0 for i in range(gnomeSize)]

        motherTemp = mother.copy()
        fatherTemp = father.copy()

        for i in range(int(gnomeSize / 2)):
            child1[gnomePosition[i]] = mother[gnomePosition[i]]
            child2[gnomePosition[i]] = father[gnomePosition[i]]
            motherTemp.remove(father[gnomePosition[i]])
            fatherTemp.remove(mother[gnomePosition[i]])

        for i in range(int(gnomeSize)):
            if child1[i] == 0:
                child1[i] = fatherTemp.pop()
            if child2[i] == 0:
                child2[i] = motherTemp.pop()

        return [child1, child2]

    # Grade the population
    def getPopulationFitness(self):
        return [(self.getFitness(x), x) for x in self.population]

    # Mutates random children with random crossover
    def mutate(self, children, mutationRate):
        number_of_mutations = int(len(children) * mutationRate)
        for _i in range(number_of_mutations):
            child = children[randint(0, len(children) - 1)]
            r1 = randint(0, len(child) - 1)
            r2 = randint(0, len(child) - 1)
            while r1 != r2:
                r2 = randint(0, len(child) - 1)

            temp = child[r1]
            child[r1] = child[r2]
            child[r2] = temp

    # Make next generation
    def next_gen(self, retain, mutationRate=0.6, randomSelect=0.6):
        graded = self.getPopulationFitness()
        graded = [x[1] for x in sorted(graded)]
        retainLength = int(len(graded) * retain)
        parents = graded[:retainLength]

        for individual in graded[retainLength:]:
            if randomSelect > random():
                parents.append(individual)

        desiredLength = len(self.population) - len(parents)
        children = []
        while len(children) < desiredLength:
            male = randint(0, len(parents) - 1)
            female = randint(0, len(parents) - 1)
            if male != female:
                children += self.makeChild(parents[female], parents[male])

        if len(children) != desiredLength:
            children.pop()

        self.mutate(children, mutationRate)

        self.population = parents
        self.population += children

    # Get one of the best in individual of current generation
    def getCurrentBest(self):
        graded = self.getPopulationFitness()
        graded.sort(key=lambda x: x[0])
        return graded[0]


if __name__ == "__main__":
    oldp = 0

    # Pretty print the sum of population fitness, difference between current and last
    # and best individual of current population
    def printFitness(e):
        global oldp
        p = [p[0] for p in e.getPopulationFitness()]
        print(f"Total: {sum(p)}, Dif: {abs(oldp - sum(p))}, Best: {e.getCurrentBest()}")
        oldp = sum(p)


    ea = EA(100)
    for i in range(100):
        ea.next_gen(0.8, 0.8)
        printFitness(ea)