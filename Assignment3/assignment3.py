from sklearn import preprocessing
import struct
import numpy as np
import random
import math

def createFeedForwardNetwork(numInputUnits, numHiddenUnits, numOutputUnits):
    network = []    
    # create hidden units layer with numInputUnits + bias randomly generated weights 
    network.append([[random.uniform(-0.5, 0.5) for _ in range((numInputUnits + 1))] for _ in range(numHiddenUnits)])    
    # create output units layer with numHiddenUnits + bias randomly generated weights 
    network.append([[random.uniform(-0.5, 0.5) for _ in range((numHiddenUnits + 1))] for _ in range(numOutputUnits)])    
    return network

def feedForwardAndBackPropogate(network, trainImagesArr, trainLabelsArr, testImagesArr, testLabelsArr, epochs, momentum, learningRate, batchSize, isRelu = False):
    for epoch in range(epochs):        
        hiddenLayerDeltaWeights = [None] * batchSize
        outputLayerDeltaWeights = [None] * batchSize
        
        avgHiddenWeightDelta = np.zeros(len(trainImagesArr[0]) + 1)
        avgOutputWtDelta = np.zeros(len(network[0]) + 1)
        
        for imgIdx in range(len(trainImagesArr)):
            image, label = trainImagesArr[imgIdx], trainLabelsArr[imgIdx]
            
            if isRelu == True:
                outputs = propogateInputForward(image, network, True)
                errorSignalsOutputLayer = [calcErrorForOutputUnitRelu(outputs[-1][i], 1.0 if label == i else 0.0) for i in range(len(outputs[-1]))]            
                hiddenLayerErrorOnOutput = np.dot(errorSignalsOutputLayer, network[1])
                hiddenLayerErrorDerivative = [1.0 if x > 0 else 0.0 for x in outputs[0]]
                errorSignalsHiddenLayer = np.multiply(hiddenLayerErrorDerivative, hiddenLayerErrorOnOutput[:-1])            
                hiddenLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(image, 1) * (np.array(errorSignalsHiddenLayer)[:, np.newaxis])
                outputLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(outputs[0], 1) * (np.array(errorSignalsOutputLayer)[:, np.newaxis])
            else :    
                outputs = propogateInputForward(image, network)
                errorSignalsOutputLayer = [calcErrorForOutputUnit(outputs[-1][i], 1.0 if label == i else 0.0) for i in range(len(outputs[-1]))]            
                hiddenLayerErrorOnOutput = np.dot(errorSignalsOutputLayer, network[1])
                hiddenLayerErrorDerivative = [getSigmoidDerivative(x) for x in outputs[0]]
                errorSignalsHiddenLayer = np.multiply(hiddenLayerErrorDerivative, hiddenLayerErrorOnOutput[:-1])            
                hiddenLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(image, 1) * (np.array(errorSignalsHiddenLayer)[:, np.newaxis])
                outputLayerDeltaWeights[imgIdx % batchSize] = learningRate * np.append(outputs[0], 1) * (np.array(errorSignalsOutputLayer)[:, np.newaxis])
            
            if (imgIdx + 1) % batchSize == 0:
                avgHiddenWeightDelta = 1.0/batchSize * np.sum(hiddenLayerDeltaWeights, axis=0) + momentum * avgHiddenWeightDelta
                avgOutputWtDelta = 1.0/batchSize * np.sum(outputLayerDeltaWeights, axis=0) + momentum * avgOutputWtDelta
                network[0] = np.add(network[0], avgHiddenWeightDelta)
                network[1] = np.add(network[1], avgOutputWtDelta)
            
            if (imgIdx == (len(trainImagesArr) - 1)/2):
                print epoch + 1, 'half'
                print predictOutputAndComputeErrors(network, trainImagesArr, trainLabelsArr, "trainingDataOutput.csv")
                print predictOutputAndComputeErrors(network, testImagesArr, testLabelsArr, "testDataOutput.csv")
                
            if (imgIdx == len(trainImagesArr) - 1):
                print epoch + 1, 'full'
                print predictOutputAndComputeErrors(network, trainImagesArr, trainLabelsArr, "trainingDataOutput.csv")
                print predictOutputAndComputeErrors(network, testImagesArr, testLabelsArr, "testDataOutput.csv")
            
def propogateInputForward(image, network, isRelu = False):
    if isRelu == True:
        hiddenLayerNetValues = np.inner(np.append(image, [1]), network[0])
        hiddenLayerVal = [max(0, output) for output in hiddenLayerNetValues]
        outputLayerNetValues = np.inner(np.append(hiddenLayerVal, [1]), network[1])
        outputLayerVal = [max(0, output) for output in outputLayerNetValues]
        return [hiddenLayerVal, outputLayerVal]
    else:
        hiddenLayerNetValues = np.inner(np.append(image, [1]), network[0])
        hiddenLayerVal = [computeSigmoid(netVal) for netVal in hiddenLayerNetValues]
        outputLayerNetValues = np.inner(np.append(hiddenLayerVal, [1]), network[1])
        outputLayerVal = [computeSigmoid(netVal) for netVal in outputLayerNetValues]
        return [hiddenLayerVal, outputLayerVal]
    
def calcErrorForOutputUnit(predictedOuput, actualOuput ):
    return getSigmoidDerivative(predictedOuput) * (actualOuput - predictedOuput)
def calcErrorForOutputUnitRelu(predictedOuput, actualOuput ):
    if predictedOuput <= 0 :
        return 0.0
    else:
        return (actualOuput - predictedOuput)

def getSigmoidDerivative(x):
    return x * (1.0 - x)
def computeSigmoid(x):
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))

def predictOutputAndComputeErrors(network, testImagesData, testImagesDataLabel, fileName):
    f=open(fileName, "a+")
    errorRates = 0.0
    meanSquareError = 0.0
    for i in range(len(testImagesData)):
        label = testImagesDataLabel[i]
        outputs = propogateInputForward(testImagesData[i], network)        
        predictedOutput = np.argmax(outputs[-1])    
        if label != predictedOutput:
            errorRates += 1            
        for j in range(len(outputs[-1])):
            meanSquareError += math.pow(outputs[-1][j], 2.0) if label != j else math.pow(1.0 - outputs[-1][j], 2.0)
    errorRates = errorRates/len(testImagesData)
    meanSquareError = meanSquareError/(2 * len(testImagesData))
    f.write(str(errorRates) + "," + str(meanSquareError) + "\r\n")
    f.close()              
    return errorRates, meanSquareError

#File Read
np.seterr(all='ignore')
trainLabelsFile = 'train-labels.idx1-ubyte'
with open(trainLabelsFile, 'rb') as labels:
        _, _ = struct.unpack(">II", labels.read(8))
        trainLabelsArr = np.fromfile(labels, dtype=np.int8)
trainImagesFile = 'train-images-pca.idx2-double'    
with open(trainImagesFile, 'rb') as images:
    _, _, dims = struct.unpack(">III", images.read(12))
    trainImagesArr = np.fromfile(images, dtype=np.dtype('float64').newbyteorder('>')).reshape(len(trainLabelsArr), dims)
trainImagesArr = preprocessing.scale(trainImagesArr) # scaling the training images data

testLabelsFile = 't10k-labels.idx1-ubyte'    
with open(testLabelsFile, 'rb') as labels:
    _, _ = struct.unpack(">II", labels.read(8))
    testLabelsArr = np.fromfile(labels, dtype=np.int8)
testImagesFile = 't10k-images-pca.idx2-double'
with open(testImagesFile, 'rb') as images:
    _, _, dims = struct.unpack(">III", images.read(12))
    testImagesArr = np.fromfile(images, dtype=np.dtype('float64').newbyteorder('>')).reshape(len(testLabelsArr), dims)
testImagesArr = preprocessing.scale(testImagesArr) # scaling the testing images data

# Initialize the 3-layered feed-forward network
numInputUnits = len(trainImagesArr[0])
numOutputUnits = 10
numHiddenUnits = 10
network = createFeedForwardNetwork(numInputUnits, numHiddenUnits, numOutputUnits)

learningRate = 0.1
batchSize = 100
momentum = 0.9
relu = False
feedForwardAndBackPropogate(network, trainImagesArr, trainLabelsArr, testImagesArr, testLabelsArr, 100, momentum, learningRate, batchSize)