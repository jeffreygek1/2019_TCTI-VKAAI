import math
import matplotlib.pyplot as plt
import numpy as np
import random

data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7])
dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
labels = []
for label in dates:
    if label < 20000301:
        labels.append('winter')
    elif 20000301 <= label < 20000601:
        labels.append('lente')
    elif 20000601 <= label < 20000901:
        labels.append('zomer')
    elif 20000901 <= label < 20001201:
        labels.append('herfst')
    else:
        labels.append('winter')

validationData = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7])
labelNames = ['winter', 'herfst', 'zomer', 'lente']

def assignToCluster(clusters, data):
    assigned = {}
    for entry in data:
        closest = str(closestCluster(entry, clusters))
        if closest in assigned:
            temp = assigned[closest]
            temp.append(entry)
            assigned[closest] = temp
        else:
            assigned[closest] = [entry]
    return assigned

def calculateIntraClusterDistance(means):
    intraClusterDistance = 0
    for centroid in means:
        distance = 0
        temp = []
        for i in centroid.strip('[]').replace(",", "").split():
            temp.append(float(i))

        for cluster in means[centroid]:
            distance = 0
            for j in range(len(temp)):
                distance += ((temp[j]-cluster[j])**2)
            distance += math.sqrt(distance)
            intraClusterDistance += distance
    return intraClusterDistance

def closestCluster(entry, clusters):
    distances = []
    for cluster in clusters:
        distance = 0
        for j in range(0, len(entry)):
            distance += ((entry[j]-cluster[j])**2)
        distances.append(math.sqrt(distance))
    lowest = sorted(distances)
    return clusters[distances.index(lowest[0])]

def getNewCentroid(data, length):
    centroid = []
    for i in range(0, length):
        temp = 0
        for j in range(0, len(data)):
            temp += data[j][i]
        mean = temp/(j+1)
        centroid.append(mean)
    return centroid

def getNewCentroids(data):
    centroids = []
    for i in data:
        centroids.append(getNewCentroid(data[i], len(data[i][0])))
    return centroids

def getKMeans(data, k):
    maxValues = getMaxValues(data)
    centroids = getRandomPoints(maxValues, k)
    assigned = assignToCluster(centroids, data)

    temp = 0
    while temp < k:
        temp = 0
        for i in assigned:
            temp+=1
        if temp < k:
            centroids = getRandomPoints(maxValues, k)
            assigned = assignToCluster(centroids, data)

    while True:
        temp = assigned
        centroids = getNewCentroids(assigned)
        assigned = assignToCluster(centroids, data)
        check = True
        for j in range(k):
            for l in range(len(assigned[list(assigned.keys())[j]])):
                try:
                    if assigned[list(assigned.keys())[j]][l].all() != temp[list(temp.keys())[j]][l].all():
                        check = False
                except IndexError:
                    check = False
        if check:
            return assigned

def getMaxValues(data):
    maxValues= []
    for i in range(0, len(data[0])):
        temp = []
        for dataEntry in data:
            temp.append(dataEntry[i])
            temp.sort()
        maxValues.append([temp[0]])
        temp.sort(reverse=True)
        maxValues[i] += [temp[0]]
    return maxValues

def getRandomPoints(values, k):
    points = []
    for i in range(0, k):
        point = []
        for value in values:
            point.append(random.randrange(value[0], value[1]))
        points.append(point)
    return points

means = getKMeans(data, 4)
counter = 1
for i in means:
    temp = []
    for j in means[i]:
        for index in range(len(data)):
            if str(j) == str(data[index]):
                temp.append(labels[index])
                break

    solution = ''
    amount = -1
    for option in labelNames:
        print(option, temp.count(option))
        if temp.count(option) > amount:
            solution = option
            amount = temp.count(option)
    print("Cluster {} is '{}'".format(counter, solution))
    counter += 1

y = []
x = []
for k in range(1, 11):
    distance = []
    for j in range(10):
        means = getKMeans(data, k)
        distance.append(calculateIntraClusterDistance(means))
    x.append(k)
    y.append(sorted(distance)[0])
    print("x {} y {}".format(x, y))
plt.plot(x, y)
plt.show()

# de beste k is k=3 want volgens de graph daar stopt de snelle afdaling het hardst