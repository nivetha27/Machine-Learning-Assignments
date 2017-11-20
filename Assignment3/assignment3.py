from sklearn import preprocessing
import struct
import numpy as np
import random
import math

def createFeedForwardNetwork(numInputUnits, numHiddenUnits, numOutputUnits):    
    # create hidden units layer with numInputUnits + bias randomly generated weights
    hiddenUnitLayer = []
    for idxHiddenUnit in range(numHiddenUnits):
        weightsFromHiddenToInput = []
        for idxInputUnit in range(numInputUnits + 1):
            weightsFromHiddenToInput.append(random.uniform(-0.5, 0.5))
        hiddenUnitLayer.append(weightsFromHiddenToInput)
    
    # create output units layer with numHiddenUnits + bias randomly generated weights
    outputUnitLayer = []
    for idxOutputUnit in range(numOutputUnits):
        weightsFromOutputToHidden = []
        for idxHiddenUnit in range(numHiddenUnits + 1):
            weightsFromOutputToHidden.append(random.uniform(-0.5, 0.5))
        outputUnitLayer.append(weightsFromOutputToHidden)

    weights = []    
    weights.append(hiddenUnitLayer)    
    weights.append(outputUnitLayer) 
    return weights

def feedForwardAndBackPropogate(network, trainImagesArr, trainLabelsArr, testImagesArr, testLabelsArr, epochs, momentum, learningRate, batchSize, isRelu = False):
    for epoch in range(epochs):        
        hiddenLayerDeltaWeights = [None] * batchSize
        outputLayerDeltaWeights = [None] * batchSize
        
        avgHiddenWeightDelta = np.zeros(len(trainImagesArr[0]) + 1)
        avgOutputWtDelta = np.zeros(len(network[0]) + 1)
        
        for imgIdx in range(len(trainImagesArr)):
            image, label = trainImagesArr[imgIdx], trainLabelsArr[imgIdx]
            
            if isRelu == True:
                hiddenLayerUnits, outputLayerUnits = propogateInputForward(image, network, True)
                errorSignalsOutputLayer = []
                for i in range(len(outputLayerUnits)):
                    if label == i:
                        errorSignalsOutputLayer.append(calcErrorForOutputUnitRelu(outputLayerUnits[i], 1))
                    else :
                        errorSignalsOutputLayer.append(calcErrorForOutputUnitRelu(outputLayerUnits[i], 0))
                hiddenLayerErrorDerivative = []
                for j in range(len(hiddenLayerUnits)):
                    if hiddenLayerUnits[j] > 0:
                        hiddenLayerErrorDerivative.append(1)
                    else :
                        hiddenLayerErrorDerivative.append(0)
                errorSignalsHiddenLayer = np.multiply(hiddenLayerErrorDerivative, np.dot(errorSignalsOutputLayer, network[1])[:-1])           
                hiddenLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(image, 1) * (np.array(errorSignalsHiddenLayer)[:, np.newaxis])
                outputLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(hiddenLayerUnits, 1) * (np.array(errorSignalsOutputLayer)[:, np.newaxis])
            else :    
                hiddenLayerUnits, outputLayerUnits = propogateInputForward(image, network)
                errorSignalsOutputLayer = []
                for i in range(len(outputLayerUnits)):
                    if label == i:
                        errorSignalsOutputLayer.append(calcErrorForOutputUnit(outputLayerUnits[i], 1))
                    else :
                        errorSignalsOutputLayer.append(calcErrorForOutputUnit(outputLayerUnits[i], 0))
                hiddenLayerErrorDerivative = []
                for j in range(len(hiddenLayerUnits)):
                    hiddenLayerErrorDerivative.append(getSigmoidDerivative(hiddenLayerUnits[j]))
                errorSignalsHiddenLayer = np.multiply(hiddenLayerErrorDerivative, np.dot(errorSignalsOutputLayer, network[1])[:-1])            
                hiddenLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(image, 1) * (np.array(errorSignalsHiddenLayer)[:, np.newaxis])
                outputLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(hiddenLayerUnits, 1) * (np.array(errorSignalsOutputLayer)[:, np.newaxis])
            
            if (imgIdx + 1) % batchSize == 0:
                avgHiddenWeightDelta = np.sum(hiddenLayerDeltaWeights, axis=0) * 1.0/batchSize + momentum * avgHiddenWeightDelta
                avgOutputWtDelta = np.sum(outputLayerDeltaWeights, axis=0) * 1.0/batchSize + momentum * avgOutputWtDelta
                network[0] = np.add(network[0], avgHiddenWeightDelta)
                network[1] = np.add(network[1], avgOutputWtDelta)
            
            if (imgIdx == (len(trainImagesArr) - 1)/2) or (imgIdx == len(trainImagesArr) - 1):
                msg = 'half epoch #' + str(epoch + 1)
                if (imgIdx == len(trainImagesArr) - 1):
                   msg = 'full epoch #' + str(epoch + 1)
                print msg
                predictOutputAndComputeErrors(network, trainImagesArr, trainLabelsArr, "trainingDataOutput.csv")
                predictOutputAndComputeErrors(network, testImagesArr, testLabelsArr, "testDataOutput.csv")
            
def propogateInputForward(image, weights, isRelu = False):
    inputLayerWithBias = np.append(image, [1])
    hiddenLayerNetValues = np.inner(inputLayerWithBias, weights[0])
    if isRelu == True:
        hiddenLayerVal = []
        for hiddenLayerNetValuesIdx in range(len(hiddenLayerNetValues)):
            hiddenLayerVal.append(max(0, hiddenLayerNetValues[hiddenLayerNetValuesIdx]))
        hiddenLayerWithBias = np.append(hiddenLayerVal, [1])
        outputLayerNetValues = np.inner(hiddenLayerWithBias, weights[1])
        outputLayerVal = []
        for outputLayerNetValuesIdx in range(len(outputLayerNetValues)):
            outputLayerVal.append(max(0, outputLayerNetValues[outputLayerNetValuesIdx]))
        return [hiddenLayerVal, outputLayerVal]    
    else:
        hiddenLayerVal = []
        for hiddenLayerNetValuesIdx in range(len(hiddenLayerNetValues)):
            hiddenLayerVal.append(computeSigmoid(hiddenLayerNetValues[hiddenLayerNetValuesIdx]))
        hiddenLayerWithBias = np.append(hiddenLayerVal, [1])
        outputLayerNetValues = np.inner(hiddenLayerWithBias, weights[1])
        outputLayerVal = []
        for outputLayerNetValuesIdx in range(len(outputLayerNetValues)):
            outputLayerVal.append(computeSigmoid(outputLayerNetValues[outputLayerNetValuesIdx]))
        return hiddenLayerVal, outputLayerVal   
    
def calcErrorForOutputUnit(predictedOuput, actualOuput ):
    return getSigmoidDerivative(predictedOuput) * (actualOuput - predictedOuput)
def calcErrorForOutputUnitRelu(predictedOuput, actualOuput ):
    if predictedOuput <= 0 :
        return 0
    else:
        return (actualOuput - predictedOuput)

def getSigmoidDerivative(x):
    return x * (1.0 - x)
def computeSigmoid(x):
    x = np.clip(x, -650, 650)
    return 1.0 / (1.0 + np.exp(-x))

def predictOutputAndComputeErrors(network, testImagesData, testImagesDataLabel, fileName):
    f=open(fileName, "a+")
    negativeCount = 0
    meanSquareError = 0
    for i in range(len(testImagesData)):
        expectedOutput = testImagesDataLabel[i]
        hiddenLayerUnits, outputLayerUnits = propogateInputForward(testImagesData[i], network)        
        predictedOutput = np.argmax(outputLayerUnits)    
        if expectedOutput != predictedOutput:
            negativeCount += 1            
        for j in range(10):
            if expectedOutput == j:
                meanSquareError += math.pow(1 - outputLayerUnits[j],2)
            else:
                meanSquareError += math.pow(0 - outputLayerUnits[j],2)
    errorRates = negativeCount * 100.0/len(testImagesData)
    meanSquareError = meanSquareError/(2 * len(testImagesData))
    f.write(str(errorRates) + "," + str(meanSquareError) + "\r\n")
    f.close()
    print errorRates, meanSquareError

#File Read
np.seterr(all='ignore')
trainLabelsFile = 'train-labels.idx1-ubyte'
with open(trainLabelsFile, 'rb') as labels:
        _, _ = struct.unpack(">II", labels.read(8))
        trainLabelsArr = np.fromfile(labels, dtype=np.uint8)
trainImagesFile = 'train-images-pca.idx2-double'    
with open(trainImagesFile, 'rb') as images:
    _, _, dims = struct.unpack(">III", images.read(12))
    trainImagesArr = np.fromfile(images, dtype=np.dtype('float64').newbyteorder('>')).reshape(len(trainLabelsArr), dims)
trainImagesArr = preprocessing.scale(trainImagesArr) # scaling the training images data

testLabelsFile = 't10k-labels.idx1-ubyte'    
with open(testLabelsFile, 'rb') as labels:
    _, _ = struct.unpack(">II", labels.read(8))
    testLabelsArr = np.fromfile(labels, dtype=np.uint8)
testImagesFile = 't10k-images-pca.idx2-double'
with open(testImagesFile, 'rb') as images:
    _, _, dims = struct.unpack(">III", images.read(12))
    testImagesArr = np.fromfile(images, dtype=np.dtype('float64').newbyteorder('>')).reshape(len(testLabelsArr), dims)
testImagesArr = preprocessing.scale(testImagesArr) # scaling the testing images data

numInputUnits = len(trainImagesArr[0])
numOutputUnits = 10
numHiddenUnits = 10
network = createFeedForwardNetwork(numInputUnits, numHiddenUnits, numOutputUnits)

learningRate = 0.5
batchSize = 100
momentum = 0.8
relu = False
feedForwardAndBackPropogate(network, trainImagesArr, trainLabelsArr, testImagesArr, testLabelsArr, 5, momentum, learningRate, batchSize, relu)
