""" 
Author: Yican Lin
Date: 2023/4/20
About code: Using information gain to decision each extending node, 
calculating entropy costs much time. And when some attributes' value
of the example are not in the tree's choosing range, it will extend 
all the next nodes and build many subtrees, the final category will 
be the largest-number one of subtrees' results.  
"""

from math import log
# import numpy.random as rd

def dataProcessing(filePath):
    strings = '' 
    try:
        file = open(filePath, 'r',encoding='utf-8')
        strings = file.read()     
        # print(strings)
    finally:
        if file:
            file.close()
    if len(strings) == 0:
        print('An error occurred in reading the data.')
        return 
    dataSet = [] 
    for line in strings.splitlines():
        line = line.strip().strip('.').split(', ')
        if len(line)==15 and '?' not in line:#Remove all the records containing '?'
            del line[13]#Remove the attributes "native-country"
            for i in range(len(line)):
                if i in [0,2,4,10,11,12]:#Process continuous value.
                    line[i] = int(line[i])
            dataSet.append(line)
    # print(dataSet)
    # print(len(dataSet)) 
    return dataSet
        
class DecisionTree:
    def __init__(self, dataSet: list[list], attributes: list[str]):
        self.dataSet = dataSet 
        self.attributes = attributes   
        self.numOfNodes = 0
        self.depths = []
        self.tree = {}

    def getNumOfNodes(self):
        if self.numOfNodes == 0:
            print('There is no node. Please run generateTree() to make a tree.')
            return 
        else:
            return self.numOfNodes

    def getMaxDepth(self):
        depths = list(set(self.depths))
        for i in range(1,len(depths)):
            for j in range(0,len(depths)-i):
                if depths[j] > depths[j+1]:
                    depths[j], depths[j+1] = depths[j+1], depths[j]
        return depths[-1]

    def getAttributeIndex(self, attribute: list[str]):
        for i in range(len(self.attributes)):
            if self.attributes[i] == attribute:
                return i     
        print('the attribute {} is not found.'.format(attribute))
        print(self.attributes)
        return None    

    def calcEntropy(self, dataSet: list[list]):
        labelCount = {}
        numOfExample = len(dataSet)
        for example in dataSet:
            oneLabel = example[-1]
            if oneLabel in labelCount.keys():
                labelCount[oneLabel] += 1
            else:
                labelCount[oneLabel] = 1
        entropy = 0
        for _, value in labelCount.items():
            pro = value/numOfExample
            entropy += -pro*log(pro,2)
        return entropy

    def splitDataSet(self, dataSet: list[list], index: int, value: str): # for discrete variables   
        subDataSet = []         
        for example in dataSet:  
            if example[index] == value:
                newExample = example[:index]
                newExample.extend(example[index+1:])
                subDataSet.append(newExample)           
        return subDataSet
    
    def splitDataSetBasedOnPoint(self, dataSet: list[list], index: int, point: float):# for continuous variables
        subDataSet1 = []
        subDataSet2 = []  
        for example in dataSet: 
            newExample = example[:index]
            newExample.extend(example[index+1:]) 
            if example[index] < point:
                subDataSet1.append(newExample)
            else:
                subDataSet2.append(newExample)
        return subDataSet1, subDataSet2

    def findBestSplitPoint(self, dataSet: list[list], index: int):# binary split
        values = []
        for line in dataSet:
            values.append(line[index])
        for i in range(1, len(values)):
            for j in range(0, len(values)-i):
                if values[j] > values[j+1]:
                    values[j], values[j+1] = values[j+1], values[j]
        if len(values) > 10:
            splitPoints = [values[0]+i*(values[-1]-values[0])/11 for i in range(1,11)]
        else:
            splitPoints = [(values[i]+values[i+1])/2 for i in range(len(values)-1)]
        entropyBefore = self.calcEntropy(dataSet)
        gains = []
        for point in splitPoints:
            splitedDataSet = self.splitDataSetBasedOnPoint(dataSet, index, point)
            newEntropy = 0
            for subDataSet in splitedDataSet:
                newEntropy += (len(subDataSet)/len(dataSet))*self.calcEntropy(subDataSet)
            gain = entropyBefore-newEntropy
            gains.append(gain)
        maxGain = 0
        bestPoint = splitPoints[0]
        for i in range(len(gains)):
            if gains[i] > maxGain:
                bestPoint = splitPoints[i]
                maxGain = gains[i] # find the max information gain
        print('Gain is {}'.format(maxGain))
        return bestPoint, maxGain     

    def calcGain(self, dataSet: list[list], index: int): 
        newEntropy = 0
        if isinstance(dataSet[0][index], str):
            values = set([example[index] for example in dataSet])
            splitedDataSet = []
            for value in values:
                subDataSet = []
                for example in dataSet:
                    if example[index] == value:
                        newExample = example[:index]
                        newExample.extend(example[index+1:])
                        subDataSet.append(newExample)
                splitedDataSet.append(subDataSet)
            for subDataSet in splitedDataSet:
                newEntropy += (len(subDataSet)/len(dataSet))*self.calcEntropy(subDataSet)
            gain = self.calcEntropy(dataSet)-newEntropy
            print('Gain is {}'.format(gain))
            return gain
        else:
            return self.findBestSplitPoint(dataSet, index)     

    def chooseAttribute(self, dataSet: list[list]):# choose the attribute with max information gain to extend the tree
        numOfAttributes = len(dataSet[0])-1
        maxGain = 0
        attributeIndex = 0
        if isinstance(dataSet[0][0], str):
            isContinuous = False
            point = None
        else:
            isContinuous = True
            point = self.calcGain(dataSet,0)[0]
        for i in range(1, numOfAttributes):
            result = self.calcGain(dataSet,i)
            if isinstance(result, tuple):
                gain = result[1]
                if gain > maxGain:
                    maxGain = gain
                    attributeIndex = i
                    point = result[0]
                    isContinuous = True
            else:
                gain = result
                if gain > maxGain:
                    maxGain = gain
                    attributeIndex = i
                    isContinuous = False
        if isContinuous == True:
            return attributeIndex, point
        return attributeIndex

    def getMostLabel(self, dataSet: list[list]): # get the label with more examples
        values = list(set([example[-1] for example in dataSet]))
        count = [0 for _ in range(len(values))]
        for i in range(len(values)):
            for example in dataSet:
                if example[-1] == values[i]:
                    count[i] += 1
        for i in range(1, len(count)):
            for j in range(0, len(values)-i):
                if count[j] > count[j+1]:
                    count[j], count[j+1] = count[j+1], count[j]
                    values[j], values[j+1] = values[j+1], values[j]
        return values[-1]

    def RecursivelySpanTree(self, dataSet: list[list], attributes: list[str], depth: int):
        print('The tree is extending nodes...')
        values = list(set([example[-1] for example in dataSet]))
        if len(values) == 1:
            self.numOfNodes += 1
            self.depths.append(depth + 1)
            return values[0]
        elif len(dataSet[0]) == 2:
            self.numOfNodes += 1
            self.depths.append(depth + 1)
            return self.getMostLabel(dataSet)
        result = self.chooseAttribute(dataSet)
        if isinstance(result, int):
            attributeIndex = result
        else:
            # print(result)
            attributeIndex = result[0]
            point = result[1]
        attribute = attributes[attributeIndex]
        self.numOfNodes += 1
        depth += 1
        print('it chooses attribute "{}"'.format(attribute))
       
        if isinstance(result, int):
            tree = {}
            tree[attribute] = {}
            del attributes[attributeIndex]
            attriValues = list(set([example[attributeIndex] for example in dataSet]))
            for attriValue in attriValues:
                tree[attribute][attriValue] = self.RecursivelySpanTree(
                    self.splitDataSet(dataSet,attributeIndex,attriValue),
                    attributes.copy(), depth)
        else:
            subDataSet1, subDataSet2= self.splitDataSetBasedOnPoint(dataSet,attributeIndex,point)
            if len(subDataSet1) != 0 and len(subDataSet2) != 0:
                tree = {}
                tree[attribute] = {}
                del attributes[attributeIndex]
                tree[attribute]['<'+str(point)] = self.RecursivelySpanTree(
                    subDataSet1, attributes.copy(), depth)
                tree[attribute]['>='+str(point)] = self.RecursivelySpanTree(
                    subDataSet2, attributes.copy(), depth)
            else:
                self.depths.append(depth + 1)
                return self.getMostLabel(dataSet)
        return tree

    def generateTree(self): # interface for users to generate tree
        print('The decision tree is generating...')
        dataSet = []
        for data in self.dataSet:
            dataSet.append(data.copy())
        attributes = self.attributes.copy()
        self.tree = self.RecursivelySpanTree(dataSet, attributes, 0)
        return self.tree

    def recursivelyClassify(self, tree: dict, data: list):
        firstAttribute = list(tree.keys())[0]
        oneDict = tree[firstAttribute]
        index = self.getAttributeIndex(firstAttribute)
        value = data[index]
        if isinstance(value, str):
            if value not in oneDict.keys(): # no value then traverse all subtrees and make integrated decision
                # print('Value is not found.')
                subTrees = []
                categories = []
                categoriesCount = []
                for key in oneDict.keys():
                    subTrees.append(oneDict[key])
                for subTree in subTrees:
                    if isinstance(subTree, dict):
                        categories.append(self.recursivelyClassify(subTree,data))
                    else:
                        categories.append(subTree)
                categoriesWithoutRepeat = list(set(categories))
                for ci in categoriesWithoutRepeat:
                    categoriesCount.append(0)
                    for cj in categories:
                        if cj == ci:
                            categoriesCount[-1] += 1
                for i in range(1,len(categoriesCount)):
                    for j in range(0, len(categoriesCount)-i):
                        if categoriesCount[j] > categoriesCount[j+1]:
                            categoriesCount[j],categoriesCount[j+1] = categoriesCount[j+1], categoriesCount[j]
                            categoriesWithoutRepeat[j],categoriesWithoutRepeat[j+1] = categoriesWithoutRepeat[j+1],categoriesWithoutRepeat[j]
                category = categoriesWithoutRepeat[-1]
                
            else:
                subTree = oneDict[value]
                if isinstance(subTree, dict):
                    category = self.recursivelyClassify(subTree,data)
                else:
                    category = subTree
        else:
            twoKeys = [key for key in oneDict.keys()]
            if twoKeys[0][0] == '<':
                point = float(twoKeys[0][1:])
                if value < point:
                    subTree = oneDict[twoKeys[0]]
                else:
                    subTree = oneDict[twoKeys[1]]
                if isinstance(subTree, dict):
                    category = self.recursivelyClassify(subTree,data)
                else:
                    category = subTree
        return category

    def classify(self, data: list): # interface for users to classify examples
        if self.tree:
            return self.recursivelyClassify(self.tree, data)
        else:
            print('Please run generateTree() to make a tree.')
            return
    
    def test(self, testDataSet: list[list]):
        correctCount = 0
        for data in testDataSet:
            categroy = self.classify(data)
            if categroy == data[-1]:
                correctCount += 1
        accuracy = correctCount/len(testDataSet)
        return accuracy

def main():
    #Generate decision tree using the train data, cosing about 7 minutes.
    dataSet = dataProcessing('data1\\adult.data')
    # dataSet = dataSet[:1000]
    attributes = ['age','workclass','fnlwgt','education','education-num',
                'marital-status','occupation','relationship','race',
                'sex','capital-gain','capital-loss','hours-per-week']
    decisionTree = DecisionTree(dataSet,attributes)
    tree = decisionTree.generateTree()
    print(tree)
    print('number of nodes: {}'.format(decisionTree.getNumOfNodes()))
    print('the largest depth: {}'.format(decisionTree.getMaxDepth()))

    #Test the classification ability of decision tree using the test data.
    print("Start testing classification...")

    accuracy = decisionTree.test(dataSet)
    print('The accuracy of decision tree on train data is {}.'.format(accuracy))
    
    testDataSet = dataProcessing('data1\\adult.test')
    accuracy = decisionTree.test(testDataSet)
    print('The accuracy of decision tree on test data is {}.'.format(accuracy))#0.8116865869853918

    #################################################################################################
    #    Integrate multiple decision trees
    #################################################################################################
    """ dataSet = dataProcessing('data1\\adult.data')
    attributes = ['age','workclass','fnlwgt','education','education-num',
                'marital-status','occupation','relationship','race',
                'sex','capital-gain','capital-loss','hours-per-week']
    testDataSet = dataProcessing('data1\\adult.test')
    numOfTrees = 25
    #numOfTrees, trainAccuracy, testAccuracy
    #10, 0.8547510112061535, 0.8334661354581673
    #20, 0.8479875339831576, 0.8361221779548472
    #24, 0.8456667329752668, 0.8349269588313413
    #25, 0.8482196140839466, 0.8377822045152723
    #26, 0.8446389496717724, 0.8354581673306772
    #27, 0.8470260592798886, 0.8365205843293493
    #30, 0.8432133147669253, 0.8349269588313413
    #40, 0.8394668788541874, 0.8332669322709163
    #50, 0.8360519859425768, 0.8306108897742364
    trees = []
    numOfExamples = len(dataSet)//numOfTrees
    print(numOfExamples)

    for i in range(numOfTrees):
        subDataSet = dataSet[i*numOfExamples:(i+1)*numOfExamples]
        decisionTree = DecisionTree(subDataSet,attributes)
        print('------------------------tree{}, start--------------------------'.format(i))
        print(i*numOfExamples,(i+1)*numOfExamples-1)
        decisionTree.generateTree()
        trees.append(decisionTree)
        print('number of nodes: {}'.format(decisionTree.getNumOfNodes()))
        print('the largest depth: {}'.format(decisionTree.getMaxDepth()))
        print('------------------------tree{}, end--------------------------'.format(i))

    #Test the classification ability of decision tree using the test data.
    print("Start testing classification...")
    def test(testDataSet):
        correctCount = 0
        for data in testDataSet:
            categories = []
            categoriesCount = []
            for tree in trees:
                category = tree.classify(data)
                categories.append(category)
            categoriesWithoutRepeat = list(set(categories))
            for ci in categoriesWithoutRepeat:
                categoriesCount.append(0)
                for cj in categories:
                    if cj == ci:
                        categoriesCount[-1] += 1
            for i in range(1,len(categoriesCount)):
                for j in range(0, len(categoriesCount)-i):
                    if categoriesCount[j] > categoriesCount[j+1]:
                        categoriesCount[j],categoriesCount[j+1] = categoriesCount[j+1], categoriesCount[j]
                        categoriesWithoutRepeat[j],categoriesWithoutRepeat[j+1] = categoriesWithoutRepeat[j+1],categoriesWithoutRepeat[j]
            category = categoriesWithoutRepeat[-1]
            if category == data[-1]:
                correctCount += 1
        accuracy = correctCount/len(testDataSet)
        return accuracy

    accuracy = test(dataSet)
    print('The accuracy of decision tree on train data is {}.'.format(accuracy))
    
    accuracy = test(testDataSet)
    print('The accuracy of decision tree on test data is {}.'.format(accuracy)) """

if __name__=="__main__":
    main()