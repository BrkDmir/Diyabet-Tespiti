# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:09:30 2022

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diabetes.csv")


# Outcome = 1 --> Diabet
# Outcome = 0 --> Healthy

diabet = data[data.Outcome == 1]
healthy = data[data.Outcome == 0]

# Scatter plot of dataset

plt.scatter(healthy.Age,healthy.Glucose,color="green",label="Healthy",alpha=0.8)
plt.scatter(diabet.Age,diabet.Glucose,color="red",label="Sick",alpha=0.8)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()


y = data.Outcome.values
xRawData = data.drop(["Outcome"],axis=1)


# Normalization Process
# Normalization = ( x - min(x) ) / ( max(x) - min(x) )
x = (xRawData - np.min(xRawData)) / (np.max(xRawData) - np.min(xRawData))
print(x)
# Splitting train and test groups


xTrain , xTest , yTrain , yTest = train_test_split(x,y,test_size=0.2,random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain,yTrain)
prediction = knn.predict(xTest)
print("\n\n")
print("For K=3, Test Results: ",knn.score(xTest,yTest))
print("\n\n")
counter = 1
for k in range(1,11):
    knnNew = KNeighborsClassifier(n_neighbors = k)
    knnNew.fit(xTrain,yTrain)
    print(counter," ","Accuracy Rate: %",knnNew.score(xTest,yTest)*100)
    counter +=1
    
    
# Prediction for new data

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
sc.fit_transform(xRawData)

newPrediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
print("\n\n\********************\n")  
print("New Prediction:",newPrediction[0])
print("\n\********************")






























