from scipy.spatial import distance
import numpy as np


def loadCsvToNumpy(filename):
    return np.genfromtxt(filename, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={
        5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})


def generateLabelsData(dates):
    labels = []
    for label in dates:
        if label < 20000301:
            labels.append("winter")
        elif 20000301 <= label < 20000601:
            labels.append("lente")
        elif 20000601 <= label < 20000901:
            labels.append("zomer")
        elif 20000901 <= label < 20001201:
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")
    return labels


def generateLabelsValidation(dates):
    labels = []
    for label in dates:
        if label < 20010301:
            labels.append("winter")
        elif 20010301 <= label < 20010601:
            labels.append("lente")
        elif 20010601 <= label < 20010901:
            labels.append("zomer")
        elif 20010901 <= label < 20011201:
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")
    return labels


def k_NearestNeighbours(data: list, testpoint, datalabels: list, k: int) -> str:
    distances = []
    nearestIndex = [0]*k

    for i in range(0, len(data)):
        distances.append([distance.euclidean(testpoint, data[i]), i])

    distances.sort()

    for i in range(0, k):
        nearestIndex[i] = distances[i][1]

    return mostOccurringLabel(nearestIndex, datalabels)

def getStringOfLabelPostion(position: int, label: list) -> str:
    return label[position]


def mostOccurringLabel(arr: list, labels: list) -> str:
    countDict = dict()
    mostOccuringString = ""
    mostOccoringNumber = 0
    for position in arr:
        labelString = getStringOfLabelPostion(position, labels)
        if labelString in countDict.keys():
            countDict[labelString] += 1
        else:
            countDict[labelString] = 1
        if countDict[labelString] > mostOccoringNumber:
            mostOccuringString = labelString
            mostOccoringNumber = countDict[labelString]
    return mostOccuringString


def main():
    data = loadCsvToNumpy('dataset1.csv')
    dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])
    labels = generateLabelsData(dates)

    validationData = loadCsvToNumpy('validation1.csv')
    validationDates = np.genfromtxt(
        "validation1.csv", delimiter=";", usecols=[0])
    validationLabels = generateLabelsValidation(validationDates)

    daysData = loadCsvToNumpy('days.csv')

    bestK = [0, 0]
    for k in range(3, 99):
        correct = 0
        false = 0

        for i in range(0, len(validationData)):
            if(k_NearestNeighbours(data, validationData[i], labels, k) == validationLabels[i]):
                correct += 1
            else:
                false += 1
        percentage = [((correct/(correct+false))*100), k]
        if percentage[0] > bestK[0]:
            bestK = percentage
        print('K: %d percentage: %d' %(k, percentage[0]))
    print('Best k: %d \nPercentage correct: %d%% ' % (bestK[1], bestK[0]))
    for item in daysData:
        print(k_NearestNeighbours(data, item, labels, bestK[1]))


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
