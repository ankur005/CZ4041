from sklearn.ensemble import RandomForestRegressor
from utility import outDir, dataDir
from generateDataSets import getFinalTrainAndTestSet, inverseLogTransform
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trainSet, testSet = getFinalTrainAndTestSet()
trainSet = trainSet.drop('tube_assembly_id', axis = 1)
trainLogCost = trainSet.pop('log_cost')
x_train = trainSet
y_train = trainLogCost

x_test = testSet
x_test = x_test.drop('tube_assembly_id', axis = 1)
x_test = x_test.drop('id', axis=1)

rf = RandomForestRegressor(n_estimators=100, max_features=0.9, n_jobs=-1)
rf = rf.fit(x_train, y_train)

trainPred = rf.predict(x_train)
rmsle = mean_squared_error(y_train, trainPred)
print "RMSLE Train: ",rmsle

traindf = pd.DataFrame()
traindf['cost'] = inverseLogTransform(trainPred)
traindf['id'] = traindf.index + 1
file = open('train_cost_90perc_features.csv','w')
traindf.to_csv(file)

file = open('test_cost_90perc_features.csv','w')
testdf = pd.DataFrame()
testdf['cost'] = inverseLogTransform(rf.predict(x_test))
testdf['id'] = testdf.index + 1
testdf.to_csv(file)