import random, math
import numpy as np

class Neuron:
    def __init__(self, values, learningRate = 0.1):
        self.weights = []
        self.sum = 0
        self.bias = -1
        self.activation = 0
        self.learningRate = learningRate
        self.values = values + [self.bias]
        for i in self.values:
            self.weights.append(random.uniform(-2, 2))

    def calcError(self, error = None, expectation = None):
        if error != None and expectation == None:
            self.error = self.sigmoidDerivative(self.sum) * error
        elif expectation != None and error == None:
            self.error = self.sigmoidDerivative(self.sum) * (expectation - self.activation)
        else:
            print("Error = " + str(error) + ". Expectated = " + str(expectation))
            exit()
        return self.error

    def calcNestedError(self, errors):
        for i in range(len(self.values)):
            if type(self.values[i]) is Neuron:
                self.values[i].calcError(error=errors[i])

    def getActivation(self):
        self.activation = self.sigmoid(self.getSum())
        return self.activation

    def getError(self):
        return self.error

    def getErrorPerLayer(self):
        errors = []
        for i in range(len(self.values)):
            if type(self.values[i]) is Neuron:
                errors.append(self.error * self.weights[i])
        return errors

    def getSum(self):
        self.sum = 0
        for i in range(len(self.values)):
            if type(self.values[i]) is Neuron:
                self.sum += (self.values[i].getActivation() * self.weights[i])
            else:
                self.sum += self.values[i] * self.weights[i]
        return self.sum

    def setValues(self, values):
        self.values = values + [self.bias]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def update(self, expected):
        weightDifferences = []
        for x in self.values:
            delta = x * self.learningRate * self.sigmoidDerivative(self.getSum()) * (expected - self.getActivation())
            weightDifferences.append(delta)
        for i in range(len(weightDifferences)):
            self.weights[i] = self.weights[i] + weightDifferences[i]

    def updateWeights(self):
        for i in range(len(self.values)):
            if type(self.values[i]) is Neuron:
                self.weights[i] += (self.learningRate * self.values[i].activation * self.error)
            else:
                self.weights[i] += (self.learningRate * self.values[i] * self.error)

class Network:
    def __init__(self, network):
        self.network = list(reversed(network))
        self.outputError = []

    def __str__(self):
        network = list(reversed(self.network))
        string = ""
        for index in range(len(self.network)):
            string += "Layer " + str(index+1) + "has " + str(len(network[index])) + " neurons.\n"
        string += str(len(self.network)) + " layers"
        return string

    def calcNetworkError(self, numbers):
        for i in range(len(self.network)):
            if i == 0:
                for j in range(len(self.network[i])):
                    self.network[i][j].calcError(expectation=numbers[j])
            else:
                error = []
                for entry in self.network[i - 1]:
                    if error == []:
                        error = entry.getErrorPerLayer()
                    else:
                        tmp = error
                        error = [x + y for x, y in zip(tmp, entry.getErrorPerLayer())]
                for entry in self.network[i - 1]:
                    entry.calcNestedError(error)

    def calcOutput(self):
        reversedNetwork = list(reversed(self.network))
        self.output = []

        for layer in reversedNetwork:
            if layer == reversedNetwork[0]:
                continue
            elif layer == reversedNetwork[-1]:
                for x in layer:
                    self.output.append(x.getActivation())
            else:
                for x in layer:
                    x.getActivation()

    def getOutput(self):
        self.calcOutput()
        return self.output

    def update(self, numbers):
        self.calcOutput()
        self.calcNetworkError(numbers)
        for layer in self.network:
            for entry in layer:
                entry.updateWeights()

equivalents = {"Iris-setosa": [1, 0, 0],
               "Iris-versicolor": [0, 1, 0],
               "Iris-virginica": [0, 0, 1]}

data = np.genfromtxt('bezdekIris.data.txt', delimiter=',', usecols=[0, 1, 2, 3]).tolist()
tmpTypes = np.genfromtxt('bezdekIris.data.txt', dtype=str, delimiter=',', usecols=[4])
testData = np.genfromtxt('bezdekIris.testData.txt', delimiter=',', usecols=[0, 1, 2, 3]).tolist()
testTmpTypes = np.genfromtxt('bezdekIris.testData.txt', dtype=str, delimiter=',', usecols=[4])

types = []
for flowerType in tmpTypes:
    types.append(equivalents[flowerType])

testTypes = []
for flowerType in testTmpTypes:
    testTypes.append(equivalents[flowerType])
#Making 3 layers, layer 1 is the input layer, layer 2 is a hidden layer
layer_1_1 = Neuron([0, 0, 0, 0])
layer_1_2 = Neuron([0, 0, 0, 0])
layer_1_3 = Neuron([0, 0, 0, 0])
layer_1_4 = Neuron([0, 0, 0, 0])
layer_1 = [layer_1_1, layer_1_2, layer_1_3, layer_1_4]

layer_2_1 = Neuron(layer_1)
layer_2_2 = Neuron(layer_1)
layer_2_3 = Neuron(layer_1)
layer_2_4 = Neuron(layer_1)
layer_2_5 = Neuron(layer_1)
layer_2 = [layer_2_1, layer_2_2, layer_2_3, layer_2_4, layer_2_5]

output_1 = Neuron(layer_2)
output_2 = Neuron(layer_2)
output_3 = Neuron(layer_2)

outputLayer = [output_1, output_2, output_3]

network = Network([layer_1, layer_2, outputLayer])

##learning the NN
learningRuns = 300
for i in range(learningRuns):
    for j in range(len(data)):
        for neuron in layer_1:
            neuron.setValues(data[j])
        network.update(types[j])

amountOfHits = 0
for i in range(len(testData)):
    for neuron in layer_1:
        neuron.setValues(testData[i])
    networkOutput = network.getOutput()
    if testTypes[i].index(1) == networkOutput.index(sorted(networkOutput, reverse=True)[0]):
        amountOfHits += 1
    else:
        print("Wrong: " + str(testData[i]))
        for result in range(len(networkOutput)):
            print(testTypes[i][result], networkOutput[result])

print(str(network))
print(str(amountOfHits/len(testData)*100) + "% is correct")