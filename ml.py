import pandas as pd
import numpy as np

def reshapeData(x):
    x.reshape(-1,1)

def concatenateOnes(x):
    onesData = np.ones(shape = x.shape[0]).reshape(-1,1)
    #print("OnesData : ",onesData)
    return np.concatenate((onesData,x),1)

def fitData(coefficients):
    #Check the formula for matrix method
    coefficients = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    return coefficients

if __name__ == "__main__":
    coefficients = []
    bostonData = pd.read_csv("BostonHousing.csv")
    #print(bostonData.head())
    
    # Y Label is "medv" so seperate it 
    x = bostonData.drop('medv',axis = 1).values
    y = bostonData['medv'].values
    
    # Reshape the data frame to the required format
    reshapeData(x)
    #print("After Reshaping : ",x)
    
    # Append ones at the starting , check whether your model requires appending of one before doing it
    x = concatenateOnes(x)
    #print("After Concatenation : ",x)
    
    #Fit the data to a model
    coefficients = fitData(coefficients)
    print(coefficients)
    
    
   
import pandas as pd
import numpy as np


if __name__ == "__main__":
    coefficients = []
    bostonData = pd.read_csv("BostonHousing.csv")

    x = bostonData.drop('medv',axis = 1).values
    y = bostonData['medv'].values
 
    x.reshape(-1,1)
    
    onesData = np.ones(shape = x.shape[0]).reshape(-1,1)
    x = np.concatenate((onesData,x),1)

    coefficients = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    print(coefficients)
    
    
  
 # -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:08:15 2020

@author: 
"""
#Import Modules Pandas and Numpy
import pandas as pd
import numpy as np
import operator

#Read CSV File
data = pd.read_csv("iris1.csv")

#To find Euc Distance
def ED(x1, x2, length): 
    distance = 0
    for x in range(length):
        distance += np.square(x1[x] - x2[x])
   # print(np.sqrt(distance))
    return np.sqrt(distance)

#KNN Model Definition
def knn(trainingSet, testInstance, k): 
 
    distances = {}

    #To find number of columns 
    length = testInstance.shape[1]

    for x in range(len(trainingSet)):
        dist = ED(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]

    sortdist = sorted(distances.items(), key=operator.itemgetter(1))
    #Put the index of col you wanna sort with 
    neighbors = []
    for x in range(k):
        neighbors.append(sortdist[x][0])

    Votes = {} #to get most frequent class of rows
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        #To get the last column for corresponding index 
        if response in Votes:
            Votes[response] += 1
        else:
            Votes[response] = 1
    #Appending the Variety to dict along with count
    sortvotes = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortvotes[0][0], neighbors)
    
#Input TestSet    
testSet = [[6.8, 3.4, 4.8, 2.4]]
test = pd.DataFrame(testSet)

#Different k Values
k = 6
k1 = 3

#Function Call
result,neigh = knn(data, test, k)
#result1,neigh1 = knn(data, test, k1)
print(result)






import pandas as pd
data=pd.read_csv('bayesdata.csv', header=None,delimiter=',')
del data[0]
r,c = data.shape

ones= (data[c]==1).sum()
zeros = (data[c]==0).sum()

prior_prob=list()
prior_prob.append(ones/(ones+zeros))
prior_prob.append(zeros/(ones+zeros))

O=list()
vals=dict()
yes_count=dict()
no_count=dict()
flag=0
for i in range(1,c):
        l=data[i].unique()
#Counting the yes and no in output column
        for j in l:
             temp=data.loc[data[i]==j]
             yes_count[j]=(temp[c] == 1).sum() 
             no_count[j]=(temp[c] == 0).sum()
#Checking for Zero entries
        for k in l:
            if(yes_count[k] == 0):
                flag=1
            if(no_count[k] == 0):
                flag=2
#Updating the yes and no counts
        if(flag!=0):        
            for j in l:
                 if flag==1:
                     yes_count[j]=yes_count[j]+(1/len(l))
                 if flag ==2:
                     no_count[j]=no_count[j]+(1/len(l))
# Updating Probabilites
        for k in l:
            if flag == 1:
                vals[k]=list([yes_count[k]/(ones+1),no_count[k]/zeros])
            elif flag == 2:
                vals[k]=list([yes_count[k]/(ones),no_count[k]/(zeros+1)])
            else:
                vals[k]=list([yes_count[k]/(ones),no_count[k]/zeros])
        O.append(vals)           
        flag=0
        vals={}
        yes_count={}
        no_count={}
# Testing the model
test=['R','HO','N','W']
num1=prior_prob[0]
den1=prior_prob[1]
for i in range(len(test)):
    num1*=O[i][test[i]][0]
    den1*=O[i][test[i]][1]
print("Prediction for ", test)
print('Yes:',num1/(den1+num1))










# load the iris dataset 
from sklearn.datasets import load_iris 
iris = load_iris() 

# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 

# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 

# making predictions on the testing set 
y_pred = gnb.predict(X_test) 

# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

