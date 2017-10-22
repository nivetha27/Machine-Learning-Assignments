import sys
import pprint
import random
import math
import arff
import scipy
import datetime
import time
from scipy.stats import chi2
from collections import Counter

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
    def attributeValWithMaxCount(self):
        if len(self.children) == 0 :
            return None
        maxCount = 0
        childWithMaxCount = None
        for child in self.children:
            total = child.positiveCount + child.negativeCount
            if total > maxCount:
                maxCount = total
                childWithMaxCount = child
        return childWithMaxCount.attributeValue
    def totalNodes(self):
        count = 1
        for child in self.children:
            count += child.totalNodes()
        return count
    def totalLeafNodes(self):
        if self.attributeName == True or self.attributeName == False:
            return 0
        count = 1
        for child in self.children:
            count += child.totalNodes()
        return count
    def printTree(self, depth):
        if len(self.children) == 0:
            print 'depth ', depth, ' : ', self.attributeName
            return

        for child in self.children:
            print 'depth ', depth, ' : ', self.attributeName, '-', child.attributeValue
            child.printTree(depth + 1)

class Attribute(object):
    def __init__(self, name):
        self.attributeName = name
        self.attributeValues = []
        self.positiveCount = 0
        self.negativeCount = 0
        
    def addAttributeVale(self, value):
        assert isinstance(value, AttributeValue)
        self.attributeValues.append(value)

class AttributeValue(object):
    def __init__(self, value, posCount = 0, negCount = 0, noneCount = 0):
        self.value = value
        self.positiveCount = posCount
        self.negativeCount = negCount
        self.noneCount = noneCount

def findAttributeIndex(attributes, attributeToBeFound):
    for idx in range(len(attributes)):
        if attributes[idx].attributeName == attributeToBeFound.attributeName:
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


def resetAndUpdateAttributeCount(examples, attributes, targetAttrIndex):
    #print 'getting pos neg count for  attribute started', datetime.datetime.now()
    #t0 = time.time()
    for idx in range(len(attributes)):
        attributes[idx].positiveCount = attributes[idx].negativeCount = 0
        for attributeVal in attributes[idx].attributeValues:
            positiveCount, negativeCount, sampleSize = countPositiveNegativeInSample(examples, targetAttrIndex, idx, attributeVal.value)
            attributeVal.positiveCount = positiveCount
            attributeVal.negativeCount = negativeCount
            attributes[idx].positiveCount += positiveCount
            attributes[idx].negativeCount += negativeCount
    #t1 = time.time()
    #print 'getting pos neg count for  attribute ended', datetime.datetime.now()
    #timeLapsed = t1-t0
    #print 'time taken for getting pos neg count for  attributes', timeLapsed
    return attributes

# Returns the attribute value with maximum frequency when target is TRUE and the one with maximum frequency when target is FALSE
def findMaxFreqAttrValGivenAttrName(attributeValues) :
    attributesLen = len(attributeValues)

    # if attribute has only none values then return NONE
    if attributesLen == 1 and attributeValues[0] == None:
        return None, None
    
    # Ignore the last attribute value which is NONE
    maxAttributeCountYes = 0
    maxAttributeCountNo = 0
    maxAttributeValIdxWithYes = maxAttributeValIdxWithNo = None
    for idx in range(attributesLen):
        attributesCountYes = attributeValues[idx].positiveCount
        attributesCountNo = attributeValues[idx].negativeCount
        if attributesCountYes > maxAttributeCountYes:
            maxAttributeCountYes = attributesCountYes
            maxAttributeValIdxWithYes = idx
        if attributesCountNo > maxAttributeCountNo:
            maxAttributeCountNo = attributesCountNo
            maxAttributeValIdxWithNo = idx
    
    # if maxAttributeValIdxWithYes is None replace it with maxAttributeValIdxWithNo and vice versa
    if maxAttributeValIdxWithYes == None:
        maxAttributeValIdxWithYes = maxAttributeValIdxWithNo
    if maxAttributeValIdxWithNo == None:
        maxAttributeValIdxWithNo = maxAttributeValIdxWithYes
    
    return maxAttributeValIdxWithYes, maxAttributeValIdxWithNo

def replaceNoneInExamples(examples, attributes, targetAttrIndex):
    #print 'replace missing attribute started', datetime.datetime.now()
    #t0 = time.time()
    examplesWithMissingAttrReplaced = [row[:] for row in examples]
    attrIdx = 0
    while (attrIdx < len(attributes)) :
        #print 'attributes length : ', len(attributes)
        #print 'attrIdx : ', attrIdx
        attribute = attributes[attrIdx]
        attributeValues = attribute.attributeValues
        maxAttrValIdxWithTargetTrue, maxAttrValIdxWithTargetFalse = findMaxFreqAttrValGivenAttrName(attributeValues)
        attrValMaxFreqTrue = attrValMaxFreqFalse = None
        if maxAttrValIdxWithTargetTrue != None:
            attrValMaxFreqTrue = attribute.attributeValues[maxAttrValIdxWithTargetTrue].value
        if maxAttrValIdxWithTargetFalse != None:
            attrValMaxFreqFalse = attribute.attributeValues[maxAttrValIdxWithTargetFalse].value
        deleteAttr = False # flag to check if the current attribute has only NONE in the examples and should bee removed
        if attrValMaxFreqTrue == None and attrValMaxFreqFalse == None:
            deleteAttr = True
        for i, example in enumerate(examplesWithMissingAttrReplaced):
            if deleteAttr:
                del example[attrIdx]
                del examples[i][attrIdx]
            elif example[attrIdx] == None:
                if example[targetAttrIndex] == "True":
                       attribute.positiveCount += 1
                       attributeValues[maxAttrValIdxWithTargetTrue].positiveCount += 1
                       example[attrIdx] = attrValMaxFreqTrue
                elif example[targetAttrIndex] == "False":
                       attribute.negativeCount += 1
                       attributeValues[maxAttrValIdxWithTargetFalse].negativeCount += 1
                       example[attrIdx] =  attrValMaxFreqFalse
        if deleteAttr == True:
            targetAttrIndex -= 1
            del attributes[attrIdx]
        else:
            attrIdx += 1

    #t1 = time.time()
    #print 'replace missing attribute ended', datetime.datetime.now()
    #timeLapsed = t1-t0
    #print 'time taken for replace', timeLapsed
    return examples, examplesWithMissingAttrReplaced, attributes

# This takes the attribute value that occurs max times irrespective of target attribute
def replaceNoneWithMaxAttrValIgnoreClass(examples,examplesWithMissingAttrReplaced, attributes, targetAttrIndex):
    #print 'replace missing attribute started', datetime.datetime.now()
    #t0 = time.time()
    attrIdx = 0
    while (attrIdx < len(attributes)) :
        #print 'attributes length : ', len(attributes)
        #print 'attrIdx : ', attrIdx
        attribute = attributes[attrIdx]
        attributeValues = attribute.attributeValues
        maxAttrVal = None
        maxAttrValIdx = None
        maxCount = 0
        for idx, attrVal in enumerate(attributeValues):
            if (attrVal.positiveCount + attrVal.negativeCount) > maxCount:
                maxCount = attrVal.positiveCount + attrVal.negativeCount
                maxAttrVal = attrVal
                maxAttrValIdx = idx

        deleteAttr = False # flag to check if the current attribute has only NONE in the examples and should bee removed
        if maxAttrVal == None:
            deleteAttr = True
        for i, example in enumerate(examplesWithMissingAttrReplaced):
            if deleteAttr:
                del example[attrIdx]
                del examples[i][attrIdx]
            elif example[attrIdx] == None:
                example[attrIdx] = maxAttrVal.value
                if example[targetAttrIndex] == "True":
                       attribute.positiveCount += 1
                       attributeValues[maxAttrValIdx].positiveCount += 1                       
                elif example[targetAttrIndex] == "False":
                       attribute.negativeCount += 1
                       attributeValues[maxAttrValIdx].negativeCount += 1
        if deleteAttr == True:
            targetAttrIndex -= 1
            del attributes[attrIdx]
        else:
            attrIdx += 1

    #t1 = time.time()
    #print 'replace missing attribute ended', datetime.datetime.now()
    #timeLapsed = t1-t0
    #print 'time taken for replace', timeLapsed
    return examplesWithMissingAttrReplaced, attributes

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

def PrePruneUsingChiSquare(examples, attributes, targetAttrIndex, chosenBestAttr, confidence):
    if chosenBestAttr == None:
        return True

    threshold = chi2.isf(1 - confidence, len(chosenBestAttr.attributeValues) - 1)

    chosenBestAttrIdx = findAttributeIndex(attributes, chosenBestAttr)

    # Find overall positive and negative ratio for the given attribute in the examples
    countPosInExample = attributes[targetAttrIndex].positiveCount
    countNegInExample = attributes[targetAttrIndex].negativeCount
    totalCount = countPosInExample + countNegInExample
   
    positiveRatio = negativeRatio = 0
    if (totalCount) :
        positiveRatio = countPosInExample / float(totalCount)
        negativeRatio = countNegInExample / float(totalCount)

    chisquare = 0;
    # Exclude None from the attribute value in calculation
    for attrVal in chosenBestAttr.attributeValues:
        actualPositive = actualNegative = 0
        expectedPositive = expectedNegative = 0
        actualPositive = attrVal.positiveCount
        actualNegative = attrVal.negativeCount
        expectedPos = (actualPositive + actualNegative) * positiveRatio
        expectedNeg = (actualPositive + actualNegative) * negativeRatio
        if expectedPos != 0 :
            chisquare += (actualPositive - expectedPos)**2 / float(expectedPos)
        if expectedNeg != 0 :
            chisquare += (actualNegative - expectedNeg)**2 / float(expectedNeg)
    #print "Chi Square of attributeVal", chosenBestAttr.attributeName, ":", chisquare
    if chisquare < threshold:
        return False
    return True

# Best Attribute is chosen based off gain ratio
def chooseBestAttribute(examples, attributes, targetAttIndex, sampleSize):
	# entropy for training data on target attribute play tennis
	countYes  = attributes[targetAttIndex].positiveCount
	countNo  = attributes[targetAttIndex].negativeCount

	entropy = calculateEntropy(countYes, countNo, sampleSize)

	bestGainRatio = 0
	bestAttr = None
	for idx in range(len(attributes)):
		attribute = attributes[idx]
		if idx != targetAttIndex :
			#print attribute
			gain = entropy
			split = 0
			attributeValues = attribute.attributeValues
            # exclude NONE from caluclation
			for attrVal in attributeValues:
				countTargetAttrYes = attrVal.positiveCount
				countTargetAttrNo = attrVal.negativeCount
				attrbValCountInSample = attrVal.positiveCount + attrVal.negativeCount
				attrEntropy = calculateEntropy(countTargetAttrYes, countTargetAttrNo, attrbValCountInSample)
				probabilityOfAttrValInSample = 0;
				if sampleSize > 0 :	
					probabilityOfAttrValInSample = (attrbValCountInSample)/float(sampleSize);
				partialGain = probabilityOfAttrValInSample * attrEntropy
				gain -= partialGain;
				if probabilityOfAttrValInSample != 0 :
					split -= probabilityOfAttrValInSample * math.log(probabilityOfAttrValInSample,2)
			#print "Gain of attributeVal", attribute.attributeName, ":", gain
			#print "Split of attributeVal", attribute.attributeName, ":", split
			if split != 0 :
				gainRatio = gain/split
				#print  "gainRatio of attributeVal", attribute.attributeName, ":", gainRatio
				if gainRatio > bestGainRatio:
					bestAttr = attribute
					bestGainRatio = gainRatio
	if bestAttr != None:
	    print 'chosen best attr', bestAttr.attributeName
	#    print 'chosen best attr positive count ', bestAttr.positiveCount
	#    print 'chosen best attr negative count ', bestAttr.negativeCount
	else:
	    print 'chosen best attr is None'
	return bestAttr

def printAttributesCount(examples, attributes):
    for idx, attr in enumerate(attributes):
        print attr.attributeName, ': ', Counter([example[idx] for example in examples]).most_common()

def buildDTUsingID3(examples, attributes, targetAttribute, confidence):
    sampleSize = len(examples)
    print 'sample size :', sampleSize
    targetAttrIndex = findAttributeIndex(attributes, targetAttribute)
    root = DecisionTreeNode()

    attributes = resetAndUpdateAttributeCount(examples, attributes, targetAttrIndex)


    positiveCount = attributes[targetAttrIndex].positiveCount
    negativeCount = attributes[targetAttrIndex].negativeCount    
    root.assignCounts(positiveCount, negativeCount)
        
    # If all Examples are positive, Return the single-node tree Root, with label = + 
    if negativeCount == 0:
        print 'all Examples are positive, Return the single-node tree Root, with label = + '
        root.assignLabel("True")
        return root
        
    # If all Examples are negative, Return the single-node tree Root, with label = - 
    if positiveCount == 0:
        print ' all Examples are negative, Return the single-node tree Root, with label = -'
        root.assignLabel("False")
        return root
        
    # Attributes is empty - assign most common value of targetAttribute in examples
    mostCommonValOfTargetInExmaple = "False"
    if positiveCount > negativeCount:
        mostCommonValOfTargetInExmaple = "True"
    if len(attributes) == 0 or len(attributes) == 1:
        print 'Attributes is empty - assign most common value (',mostCommonValOfTargetInExmaple,') of targetAttribute in examples'
        root.assignLabel(mostCommonValOfTargetInExmaple)
        return root

    # Replace Missing Attribute in an example with the frequent attribute value corresponding to the example target class
    examplesWithMissingAttrReplaced = [row[:] for row in examples]
    attributesWithMissingReplaced = attributes[:]
    #examples, examplesWithMissingAttrReplaced, attributesWithMissingReplaced = replaceNoneInExamples(examplesWithMissingAttrReplaced, attributesWithMissingReplaced, targetAttrIndex)
    #printAttributesCount(examplesWithMissingAttrReplaced, attributesWithMissingReplaced)
    examplesWithMissingAttrReplaced, attributesWithMissingReplaced = replaceNoneWithMaxAttrValIgnoreClass(examples, examplesWithMissingAttrReplaced, attributesWithMissingReplaced, targetAttrIndex)
    #printAttributesCount(examplesWithMissingAttrReplaced, attributesWithMissingReplaced)

    targetAttrIndex = findAttributeIndex(attributesWithMissingReplaced, targetAttribute)
    # Choose the best attribute based on the given setting
    bestAttr = chooseBestAttribute(examplesWithMissingAttrReplaced, attributesWithMissingReplaced, targetAttrIndex, sampleSize)
    
    # Determine if best attribute should be pre-pruned or not
    useCurBestAttrToFormTree = PrePruneUsingChiSquare(examplesWithMissingAttrReplaced, attributesWithMissingReplaced, targetAttrIndex, bestAttr, confidence)
    if useCurBestAttrToFormTree == False :
        bestAttr = None

    # If we did not find an attribute, assign most common value of targetAttribute in examples
    if bestAttr is None:
        print 'Best attribute is empty - assign most common value (',mostCommonValOfTargetInExmaple,') of targetAttribute in examples'
        root.assignLabel(mostCommonValOfTargetInExmaple)
        return root

    bestAttrIndex = findAttributeIndex(attributesWithMissingReplaced, bestAttr)
    
    # Assign label as attribute name
    root.assignLabel(bestAttr.attributeName)
        
    # Add a branch for each possible value
    for bestAttrVal in bestAttr.attributeValues:
        subSetWithAttrVal = []
        for exampleIdx in range(len(examplesWithMissingAttrReplaced)):
            if examplesWithMissingAttrReplaced[exampleIdx][bestAttrIndex] == bestAttrVal.value:
                subSetWithAttrVal.append(examples[exampleIdx][:])
        if len(subSetWithAttrVal) == 0:
            child = DecisionTreeNode('root', bestAttrVal.value, None)
            root.addChild(child)
            #print "Assigned ", mostCommonValOfTargetInExmaple
            child.assignLabel(mostCommonValOfTargetInExmaple)
        else:
            newAttributesList = attributesWithMissingReplaced[:]
            del newAttributesList[bestAttrIndex]
            # in the subset remove the column that has the best attribute
            for set in subSetWithAttrVal:
                del set[bestAttrIndex]
            #print 'calling ID3 for current best attribute(', bestAttr.attributeName,') with value ', bestAttrVal.value
            child = buildDTUsingID3(subSetWithAttrVal, newAttributesList, targetAttribute, confidence)
            root.addChild(child)
            child.assignValue(bestAttrVal.value)
    if len(root.children) == 0:
        print "Attribute node does not have any child"
    
    return root

def getClassificationOnData(tree, testDataSet, attributes, targetAttribute):
    if tree.attributeName == "True":
        return "True"
    
    if tree.attributeName == "False":
        return "False"

    for attributeIdx, attribute in enumerate(attributes):
        attributeName = attribute.attributeName
        if tree.attributeName == attributeName:
            if testDataSet[attributeIdx] == None:
                testDataSet[attributeIdx] = tree.attributeValWithMaxCount()
            for child in tree.children:
                if child.attributeValue == testDataSet[attributeIdx]:
                    return getClassificationOnData(child, testDataSet, attributes, targetAttribute)
    return tree.attributeName;                

def determineAccuracy(tree, testDataSet, attributes, targetAttribute):
    
    totalCount = len(testDataSet)
    positive = good = 0
    targetAttrIndex = findAttributeIndex(attributes, targetAttribute)
    
    expectedPositives = expectedNegatives = predictedPositives = preditedNegatives = 0
    correctlyPredictedPositives = correctlyPredictedNegatives = 0
    incorrectlyPredictedPositives = incorrectlyPredictedNegatives = 0
    
    # Predict for every test case
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
            else:
                correctlyPredictedNegatives += 1
        else:
            if expected == "True":
                incorrectlyPredictedNegatives += 1
            else:
                incorrectlyPredictedPositives += 1
        if actual == "True" or actual == "False":
            good += 1
    goodEval = 0
    if totalCount != 0:
        goodEval = good/float(totalCount)
    precision = 0
    if predictedPositives != 0:
        precision = correctlyPredictedPositives/float(predictedPositives)
    recall = 0
    if expectedPositives != 0:
        recall = correctlyPredictedPositives/float(expectedPositives)
    print 'expectedPositive : ', expectedPositives
    print 'expectedNegatives : ', expectedNegatives
    print 'predictedPositives : ', predictedPositives
    print 'preditedNegatives : ', preditedNegatives
    print 'correctlyPredictedPositives : ', correctlyPredictedPositives
    print 'correctlyPredictedPositives : ', correctlyPredictedNegatives
    print 'incorrectlyPredictedPositives : ', incorrectlyPredictedPositives
    print 'incorrectlyPredictedNegatives : ', incorrectlyPredictedNegatives
    print("Good eval:", goodEval*100)   
    print("Precision: ", precision * 100)
    print("Recall: ", recall)
    
    # Return the accuracy
    accuracy = positive*100/float(totalCount)
    print("Accuracy:", accuracy)
    return accuracy



# Load training data
#fileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\training_subset300.arff"
fileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\training_subsetD.arff"
#fileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\weather_data_subset.arff"
print 'opening file ', fileName
data = arff.load(open(fileName, 'rb'))
print 'opened file ', fileName
samples = [row[:] for row in data['data']]
attributes = []
# The attributes can just be None in the given set, therefore explicitly adding None as an attribute value.
for attribute in data['attributes']:
    attributeObject = Attribute(attribute[0])
    for attrVal in attribute[1]:
        #if (attrVal != 'NULL'):
        #    attrValObject = AttributeValue(attrVal)
        #    attributeObject.addAttributeVale(attrValObject)
        attrValObject = AttributeValue(attrVal)
        attributeObject.addAttributeVale(attrValObject)
    attributes.append(attributeObject)

targetAttribute = attributes[-1]

# Load test data
testFileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\testingD.arff"
#testFileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\testing300.arff"
#testFileName = "C:\\Users\\nsathya\\Documents\\Visual Studio 2015\\Projects\\PythonApplication1\\PythonApplication1\\weather_test_data_subset.arff"
print 'opening file ', testFileName
testData = arff.load(open(testFileName))
print 'opened file ', testFileName

def addAllPositiveSamples(samples) :
    examples =  [row[:] for row in samples]
    for example in examples:
        if example[-1] == "True":
            samples.append(example)


confidences = [0.95]#,.5,.8,.95,.99]
for confidence in confidences:
    print "building tree with confidence : ", confidence 
    #tree = buildDTUsingID3(samples[1:1000], attributes, targetAttribute, confidence)
    #addAllPositiveSamples(samples)
    tree = buildDTUsingID3(samples, attributes, targetAttribute, confidence)
    attr = tree.attributeValWithMaxCount()
    #accuracy = determineAccuracy(tree, testData['data'][1:100], attributes, targetAttribute)
    
    print 'metrics on test data'
    accuracy = determineAccuracy(tree, testData['data'], attributes, targetAttribute)
    print 'metrics on traning data'
    accuracy = determineAccuracy(tree, data['data'], attributes, targetAttribute)
    print 'Node in tree', tree.totalNodes()
    #print 'decision nodes in tree', tree.totalDecisionNodes()
    print tree.printTree(0)
# shouldPrePrune = PrePruneUsingChiSquare( samples, attributes, targetAttribute, attributes[1], 0.95)
# bestAttr = chooseBestAttribute(samples, attributes, targetAttribute)
print "End of DT"