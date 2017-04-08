from sklearn.ensemble import RandomForestRegressor
from utility import outDir, dataDir
from generateDataSets import getFinalTrainAndTestSet, inverseLogTransform
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

print list(x_train)
print list(x_test)
rf = RandomForestRegressor()
rf = rf.fit(x_train, y_train)
result = rf.predict(x_test)
print result
print inverseLogTransform(result)