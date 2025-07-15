""" 
Author: Yican Lin
Date: 2023/4/23
About code: Using SMO to update the value of alpha and b. Since the result is not very good,
I change the way to select the index of alpha from judging KKT condition to randomly selecting.
And I design a argument called goBack, to allow arguments go back to last state, when the 
updating makes the accuracy fall down more than 0.1 and goBack is lass than 1. And when the 
arguments updates, goBack will be reduced by 1. And I found 20 iterations is enough, because 
the accurary can be hardly more than 80%, and the best one can almost reach in 20 iterations.   
Before training, I do some data normalization to make data satisfy that the mean is 0 and the
variance is 1. Also the discrete variables are mapped to numbers and then do normalization.
"""
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt 

def dataProcessing(filePath, cut: bool = False):
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
    
    for i in [1,3,5,6,7,8,9]:
        values = []
        index = -1
        for j in range(len(dataSet)):    
            if dataSet[j][i] not in values:
                values.append(dataSet[j][i])
                index += 1
                dataSet[j][i] = index
            else:
                dataSet[j][i] = values.index(dataSet[j][i])   
    for data in dataSet:
        if data[-1] == '<=50K':
            data[-1] = -1
        else:
            data[-1] = 1

    indexes = []
    for i in range(len(dataSet)):
        if dataSet[i][-1] == -1:
            indexes.append(i)
    if cut:
        count = 0
        for i in range(len(indexes)):
            cond = np.random.random() < 0.7
            if cond:
                del dataSet[indexes[i]-count]
                count += 1

    features = [dataSet[i][:-1] for i in range(len(dataSet))]
    labels = [dataSet[i][-1] for i in range(len(dataSet))]
    
    values = [[] for _ in range(len(dataSet[0])-1)]
    for i in range(len(dataSet[0])-1):
        for j in range(len(dataSet)):
                values[i].append(features[j][i])

    means = []
    stds = []
    for v in values:
        sum = 0
        for i in range(len(v)):
            sum += v[i]
        mean = sum/len(v)
        sum = 0
        for i in range(len(v)):
            sum += (v[i]-mean)**2
        std = (sum/len(v))**0.5
        means.append(mean)
        stds.append(std)      

    for data in features:
        for i in range(len(data)):
            data[i] = (data[i]-means[i])/stds[i] #data normalization

    features = np.array(features)
    labels = np.array(labels)
    print(features.shape)#(30162, 13)
    print(labels.shape)#(30162,)

    for i in range(10):
        print(features[i])

    return features, labels

class SVM:
    def __init__(self, features: np.array, labels: np.array, maxIter: int = 100, kernel: str = 'linear'):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.maxIter = maxIter
        self.kernel = kernel
        self.b = 0.0
        self.C = 1.0#penalty factor
        self.alpha = np.random.rand(self.m)*self.C
        # self.alpha = 0.5*np.ones(self.m)
        self.bestAlpha = self.alpha
        self.bestB = self.b
        self.bestAccuracyOnTrainData = 0

    def getTestDataSet(self, testFeatures: np.array, testLabels: np.array):
        self.testX = testFeatures
        self.testY = testLabels

    def kernelFunction(self, x1, x2, sigma = 1.0): # gauss kernel is slow so I choose linear kernel when running
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'ploy':
            return (np.dot(x1, x2)+1)**2
        elif self.kernel == 'gauss':
            return np.exp(-np.linalg.norm(x1-x2)**2/(2*sigma**2))
        elif self.kernel == 'laplace':
            return np.exp(-np.linalg.norm(x1-x2)/(2*sigma))
        else:
            return 

    def predictFunction(self, x):
        return np.sum(self.alpha*self.Y*self.kernelFunction(self.X,x))+self.b

    def error(self, i):
        return self.predictFunction(self.X[i]) - self.Y[i]

    def clipAlpha(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha
    
    def randIndex(self): # choose one index randomly as the first alpha's index of SMO algorithm
        indexes = [i for i in range(self.m)]
        i = np.random.choice(indexes)
        return i
        
    def SMO(self, method: str = 'KKT'):# SMO algorithm with a few new tips 
        print('Start training...')
        alpha_i, alpha_j, index_i, index_j = None, None, None, None
        E = np.array([self.error(i) for i in range(self.m)])
        self.accuracies = []
        self.testAccuracies = []
        goBack = 0 # goback makes the parameter space extend gradually
        # nochange = []
        for k in range(self.maxIter):
            print('iteration {}:'.format(k))
            if method == 'KKT': # KKT may not choose the most suitable argument
                flag = 0
                for i in range(self.m):
                    if 0 < self.alpha[i] < self.C:
                        if self.Y[i]*self.predictFunction(self.X[i]) != 1:
                            index_i = i
                            flag = 1
                            break
                if flag == 0:
                    for i in range(self.m):
                        if self.alpha[i] == 0:
                            if self.Y[i]*self.predictFunction(self.X[i]) < 1:
                                index_i = i
                                flag = 1
                                break
                        elif self.alpha[i] == self.C:
                            if self.Y[i]*self.predictFunction(self.X[i]) > 1:
                                index_i = i
                                flag = 1
                                break
                if flag == 0:
                    break
            elif method == 'random':
                index_i = self.randIndex()
            else:
                return
            # nochange.clear()
            alpha_i = self.alpha[index_i]
            Ei = E[index_i]
            if Ei >= 0:
                index_j = np.argmin(E)
            else:
                index_j = np.argmax(E)
            alpha_j = self.alpha[index_j]
            Ej = E[index_j]

            Xi, Xj, Yi, Yj = self.X[index_i], self.X[index_j], self.Y[index_i], self.Y[index_j]
            if Yi != Yj:
                L = np.max([0, alpha_j - alpha_i])
                H = np.min([self.C, self.C + alpha_j - alpha_i])
            else:
                L = np.max([0, alpha_i + alpha_j - self.C])
                H = np.min([self.C, alpha_i + alpha_j])

            Kii = self.kernelFunction(Xi, Xi)
            Kjj = self.kernelFunction(Xj, Xj)
            Kij = self.kernelFunction(Xi, Xj)
            eta = Kii + Kjj - 2*Kij
            
            if eta <= 0:
                continue

            alpha_j_unc = alpha_j + Yj*(Ei - Ej)/eta
            alpha_j_new = self.clipAlpha(alpha_j_unc, L, H)
            alpha_i_new = alpha_i + Yi*Yj*(alpha_j-alpha_j_new)

            b_old = self.b
            bi_new = -Ei - Yi*Kii*(alpha_i_new - alpha_i) - Yj*Kij*(alpha_j_new - alpha_j) + self.b
            bj_new = -Ej - Yi*Kij*(alpha_i_new - alpha_i) - Yj*Kjj*(alpha_j_new - alpha_j) + self.b
            if 0 < alpha_i_new < self.C:
                b_new = bi_new
            elif 0 < alpha_j_new < self.C:
                b_new = bj_new
            else:
                b_new = (bi_new + bj_new)/2
            
            if alpha_j != alpha_j_new:
                print('change alpha')
            # else:
            #     nochange.append(index_i)

            # update arguments
            self.alpha[index_i] = alpha_i_new
            self.alpha[index_j] = alpha_j_new
            self.b = b_new
            accuracy = self.test(self.X, self.Y)#  it will choose the best arguments base on train data rather than test data
            testAccuracy = self.test(self.testX, self.testY) # here just to see the test reslt but it won't be used to help training. 

            if len(self.accuracies)!=0: 
                if accuracy - self.accuracies[-1] <= -0.1 and goBack < 1:# give chance to go back to old 
                    self.alpha[index_i] = alpha_i
                    self.alpha[index_j] = alpha_j
                    self.b = b_old
                    goBack += 1
                else:
                    goBack -= 1
                    self.accuracies.append(accuracy)
                    if accuracy > self.bestAccuracyOnTrainData:# the best arguments are the ones with the highest accuracy on train data
                        self.bestAccuracyOnTrainData = accuracy
                        self.bestAlpha = self.alpha
                        self.bestB = self.b
                    self.testAccuracies.append(testAccuracy)
                    print('The accuracy of support vector mechine on train data is {}.'.format(accuracy))
                    print('The accuracy of support vector mechine on test data is {}.'.format(testAccuracy))
                    E = np.array([self.error(i) for i in range(self.m)])
            else:
                self.accuracies.append(accuracy)
                self.testAccuracies.append(testAccuracy)
                print('The accuracy of support vector mechine on train data is {}.'.format(accuracy))
                print('The accuracy of support vector mechine on test data is {}.'.format(testAccuracy))
                E = np.array([self.error(i) for i in range(self.m)])
                   

    def predict(self, x):
        pred = self.predictFunction(x)
        return 1 if pred > 0 else -1
    
    def test(self, testFeatures: np.array, testLabels: np.array): # for inner test while training
        correctCount = 0
        # count1 = 0
        # realCount = 0
        for i in range(testFeatures.shape[0]):
            if self.predict(testFeatures[i]) == testLabels[i]:
                correctCount += 1
            # if self.predict(testFeatures[i]) == 1:
            #     count1 += 1
            # if testLabels[i] == 1:
            #     realCount += 1
        correctRate = correctCount/testFeatures.shape[0]
        # rate = count1/testFeatures.shape[0]
        # rate2 = realCount/testFeatures.shape[0]
        # print('The accuracy of support vector mechine is {}.'.format(correctRate))
        # print('the rate of predict 1 is {}'.format(rate))
        # print('the rate of real 1 is {}'.format(rate2))
        return correctRate
    
    def testWithBestArg(self, testFeatures: np.array, testLabels: np.array): # for the final test
        correctCount = 0
        for i in range(testFeatures.shape[0]):
            result = np.sum(self.bestAlpha*self.Y*self.kernelFunction(self.X,testFeatures[i]))+self.bestB
            pred = 1 if result > 0 else -1
            if pred == testLabels[i]:
                correctCount += 1
        correctRate = correctCount/testFeatures.shape[0]
        return correctRate

    def drawLine(self): # draw the line of accuracy
        x = [i for i in range(len(self.accuracies))]
        plt.figure()
        plt.title('Accuracy of support vector mechine')
        plt.plot(x,self.accuracies,label="accuracy on train data")
        plt.plot(x,self.testAccuracies, label="accuracy on test data")
        plt.legend()
        plt.show()

def main():
    features, labels = dataProcessing('data1\\adult.data')
    # features, labels = features[:10000], labels[:10000]
    svm = SVM(features,labels,20)
    testFeatures, testLabels = dataProcessing('data1\\adult.test')
    svm.getTestDataSet(testFeatures, testLabels)
    svm.SMO(method='random')
    svm.drawLine()
    accuracy = svm.testWithBestArg(testFeatures, testLabels)
    print('final accuracy on test data is {}'.format(accuracy))#0.80066401062417
    
    
if __name__ == "__main__":
    main()