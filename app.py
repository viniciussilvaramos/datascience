from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np 

data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(data)

x = data[:, :48]
y = data[:, -1]

xTrain = x[:-100,]
yTrain = y[:-100,]
xTest = x[-100:,]
yTest = y[-100:,]

model = MultinomialNB()
model.fit(xTrain, yTrain)

print("Classification rate for NB:", model.score(xTest, yTest))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(xTrain, yTrain)

print("Classification rate for AdaBoost:", model.score(xTrain, yTrain)) 
