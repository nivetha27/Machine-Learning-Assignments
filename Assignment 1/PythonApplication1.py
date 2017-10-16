import sys
import pprint
import random
import math
import arff
import scipy
from scipy.stats import chi2

class DecisionTreeNode(object):
    def __init__(self, name='root', value = None, children=None):
        self.attributeName = name
        self.attributeValue = value
        self.children = []
        self.positiveCount = 0
        self.negativeCount = 0
        if children is not None:
            for child in children:
                self.addChild(child)
    
    def addChild(self, node):
        assert isinstance(node, DecisionTreeNode)
        self.children.append(node)
    def assignLabel(self, label):
        self.attributeName=label
    def assignValue(self, value):
        self.attributeValue = value
    def assignCounts(self, positive, negative):
        self.positiveCount = positive
        self.negativeCount = negative


def findAttributeIndex(attributes, attributeToBeFound):
    for idx in range(len(attributes)):
        if attributes[idx][0] == attributeToBeFound[0]:
            return idx

def countPositiveNegativeInSample(examples, targetAttrIndex, attributeIndex = None, attributeValue = None):
    countPositive = countNegative = 0
    sampleSize = 0
    for example in examples:
        if attributeIndex == None or attributeValue == None :
            sampleSize += 1
            if example[targetAttrIndex] == "True" :
                countPositive += 1
            else :
                countNegative +=1
        else : 
            if example[attributeIndex] == attributeValue:
                sampleSize += 1
                if example[targetAttrIndex] == "True" :
                    countPositive += 1
                else :
                    countNegative +=1
    return countPositive, countNegative, sampleSize

def calculateEntropy(countPositive, countNegative, totalCount) :
    #print "sample size", totalCount
    #print "#positive samples", countPositive
    #print "#negative samples", countNegative
    
    probabilityYes = probabilityNo = 0
    if totalCount != 0 :
        probabilityYes = countPositive/float(totalCount)
        probabilityNo = countNegative/float(totalCount)
    #print "positive sample probability", probabilityYes
    #print "negative sample probability", probabilityNo

    entropy = 0
    if probabilityYes != 0:
        entropy -= (probabilityYes * math.log(probabilityYes,2))
    if probabilityNo != 0 :
        entropy -= (probabilityNo * math.log(probabilityNo,2))
    
    #print "sample entropy", entropy
    return entropy

def PrePruneUsingChiSquare(examples, attributes, targetAttribute, chosenBestAttr, confidence):
    if chosenBestAttr == None:
        return True

    threshold = chi2.isf(1 - confidence, len(chosenBestAttr[1]) - 1)

    targetAttrIndex = findAttributeIndex(attributes, targetAttribute)
    chosenBestAttrIdx = findAttributeIndex(attributes, chosenBestAttr)

    # Find overall positive and negative ratio for the given attribute in the examples
    countPosInExample, countNegInExample, totalCount = countPositiveNegativeInSample(examples, targetAttrIndex)
   
    positiveRatio = negativeRatio = 0
    if (totalCount) :
        positiveRatio = countPosInExample / float(totalCount)
        negativeRatio = countNegInExample / float(totalCount)

    chisquare = 0;
    for attrVal in chosenBestAttr[1]:
        actualPositive = actualNegative = 0
        expectedPositive = expectedNegative = 0
        actualPositive, actualNegative, total = countPositiveNegativeInSample(examples, targetAttrIndex, chosenBestAttrIdx, attrVal)
        expectedPos = (actualPositive + actualNegative) * positiveRatio
        expectedNeg = (actualPositive + actualNegative) * negativeRatio
        if expectedPos != 0 :
            chisquare += (actualPositive - expectedPos)**2 / float(expectedPos)
        if expectedNeg != 0 :
            chisquare += (actualNegative - expectedNeg)**2 / float(expectedNeg)
    
    if chisquare < threshold:
        return False
    return True

# Best Attribute is chosen based off gain ratio
def chooseBestAttribute(trainingData, attributes, targetAttribute):
	# entropy for training data on target attribute play tennis
	targetAttIndex = findAttributeIndex(attributes, targetAttribute)
	countYes, countNo, sampleSize = countPositiveNegativeInSample(trainingData, targetAttIndex)
	entropy = calculateEntropy(countYes, countNo, sampleSize)

	bestGainRatio = 0
	bestAttr = None
	for attribute in attributes:
		if attribute != targetAttribute :
			attrIdx = findAttributeIndex(attributes, attribute)
			#print attribute
			gain = entropy
			split = 0
			attributeValues = attribute[1]
			for attrValue in attributeValues:                
				values = attrValue
				countTargetAttrYes, countTargetAttrNo, attrbValCountInSample = countPositiveNegativeInSample(trainingData, targetAttIndex, attrIdx, attrValue)
				attrEntropy = calculateEntropy(countTargetAttrYes, countTargetAttrNo, attrbValCountInSample)
				probabilityOfAttrValInSample = 0;
				if sampleSize > 0 :	
					probabilityOfAttrValInSample = (attrbValCountInSample)/float(sampleSize);
				partialGain = probabilityOfAttrValInSample * attrEntropy
				gain -= partialGain;
				if probabilityOfAttrValInSample != 0 :
					split += probabilityOfAttrValInSample * math.log(probabilityOfAttrValInSample,2)
			split *= -1;
			#print "Gain of attributeVal", attribute[0], ":", gain
			#print "Split of attributeVal", attribute[0], ":", split
			if split != 0 :
				gainRatio = gain/split	
				if gainRatio > bestGainRatio and gain != entropy:
					bestAttr = attribute
					bestGainRatio = gainRatio

	print 'chosen best attr', bestAttr
	return bestAttr

def buildDTUsingID3(examples, attributes, targetAttribute, confidence):
    targetAttrIndex = findAttributeIndex(attributes, targetAttribute)    
    positiveCount, negativeCount, sampleSize = countPositiveNegativeInSample(examples, targetAttrIndex)
    
    root = DecisionTreeNode();
    root.assignCounts(positiveCount, negativeCount)
        
    # If all Examples are positive, Return the single-node tree Root, with label = + 
    if negativeCount == 0:
        root.assignLabel("True")
        return root
        
    # If all Examples are negative, Return the single-node tree Root, with label = - 
    if positiveCount == 0:
        root.assignLabel("False")
        return root
        
    # Attributes is empty - assign most common value of targetAttribute in examples
    mostCommonValOfTargetInExmaple = "False"
    if positiveCount > negativeCount:
        mostCommonValOfTargetInExmaple = "True"
    if len(attributes) == 0 or len(attributes) == 1:
        root.assignLabel(mostCommonValOfTargetInExmaple)
        return root
        
    # Choose the best attribute based on the given setting
    bestAttr = chooseBestAttribute(examples, attributes, targetAttribute)
    
    # Determine if best attribute should be pre-pruned or not
    useCurBestAttrToFormTree = PrePruneUsingChiSquare(examples, attributes, targetAttribute, bestAttr, confidence)
    if useCurBestAttrToFormTree == False :
        bestAttr = None

    # If we did not find an attribute, assign most common value of targetAttribute in examples
    if bestAttr is None:
        root.assignLabel(mostCommonValOfTargetInExmaple)
        return root

    bestAttrIndex = findAttributeIndex(attributes, bestAttr)
    
    # Assign label as attribute name
    root.assignLabel(bestAttr[0])
        
    # Add a branch for each possible value
    for bestAttrVal in bestAttr[1]:
        subSetWithAttrVal = []
        for example in examples:
            if example[bestAttrIndex] == bestAttrVal:
                subSetWithAttrVal.append(example[:])
        if len(subSetWithAttrVal) == 0:
            child = DecisionTreeNode('root', bestAttrVal, None)
            root.addChild(child)
            #print "Assigned ", mostCommonValOfTargetInExmaple
            child.assignLabel(mostCommonValOfTargetInExmaple)
        else:
            newAttributesList = attributes[:]
            del newAttributesList[bestAttrIndex]
            # in the subset remove the column that has the best attribute
            for set in subSetWithAttrVal:
                del set[bestAttrIndex]
            child = buildDTUsingID3(subSetWithAttrVal, newAttributesList, targetAttribute, confidence)
            root.addChild(child)
            child.assignValue(bestAttrVal)
    if len(root.children) == 0:
        print "Attribute node does not have any child"
    return root

def getClassificationOnData(tree, testDataSet, attributes, targetAttribute):
    if tree.attributeName == "True":
        return "True"
    
    if tree.attributeName == "True":
        return "False"

    for attribute in attributes:
        attributeName = attribute[0]
        attributeIdx = findAttributeIndex(attributes, attribute)
        if tree.attributeName == attributeName:
            for child in tree.children:
                if child.attributeValue == testDataSet[attributeIdx]:
                    return getClassificationOnData(child, testDataSet, attributes, targetAttribute)
    return tree.attributeName;                

def determineAccuracy(tree, testDataSet, attributes, targetAttribute):
    
    totalCount = len(testDataSet)
    positive = good = 0
    targetAttrIndex = findAttributeIndex(attributes, targetAttribute)
    
    expectedPositives = expectedNegatives = predictedPositives = preditedNegatives = 0
    correctlyPredictedPositives = 0
    
    # PRedict for every test case
    for test in testDataSet:
        expected = test[targetAttrIndex]
        actual = getClassificationOnData(tree, test, attributes, targetAttribute)
        if expected == "True":
            expectedPositives += 1
        if expected == "False":
            expectedNegatives += 1
        if actual == "True":
            predictedPositives += 1
        if actual == "False":
            preditedNegatives += 1
        if expected == actual:
            positive += 1
            if expected == "True":
                correctlyPredictedPositives += 1
        if actual == "True" or actual == "False":
            good += 1
    print("Good eval:", str(float(good/totalCount)*100))    
    print("Precision: ", str(float(correctlyPredictedPositives/predictedPositives)))
    print("Recall: ", str(float(correctlyPredictedPositives/expectedPositives)))
    
    # Return the accuracy
    accuracy = float(positive/totalCount)*100
    print("Accuracy:", accuracy)
    return accuracy

# Load training data
fileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\training_subsetD.arff"
data = arff.load(open(fileName, 'rb'))
samples = data['data']
attributes = data['attributes']
targetAttribute = data['attributes'][-1]

# Load test data
testFileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\testingD.arff"
testData = arff.load(open(testFileName))

confidences = [0,.5,.8,.95,.99]
for confidence in confidences:
    print "building tree with confidence : ", confidence 
    tree = buildDTUsingID3(samples, attributes, targetAttribute, confidence)
    accuracy = determineAccuracy(tree, testData['data'], attributes, targetAttribute)
# shouldPrePrune = PrePruneUsingChiSquare( samples, attributes, targetAttribute, attributes[1], 0.95)
# bestAttr = chooseBestAttribute(samples, attributes, targetAttribute)
print "End of DT"