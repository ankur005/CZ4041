from sklearn.ensemble import RandomForestRegressor
from constants import RFRDir
from generateDataSets import getFinalTrainAndTestSet, inverseLogTransform
from sklearn.metrics import mean_squared_error
import os
import pandas as pd


# Main function to run
if __name__ == '__main__':
    # Get merged train and test dataset
    trainSet, testSet = getFinalTrainAndTestSet()
    # Modify this list to drop certain columns for experimentation
    dropcols = ['quantity', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a_forming', 'end_x_forming']
    trainSet = trainSet.drop(dropcols, axis=1)
    testSet = testSet.drop(dropcols, axis=1)
    trainSet = trainSet.drop('tube_assembly_id', axis = 1)
    trainLogCost = trainSet.pop('log_cost')
    x_train = trainSet
    y_train = trainLogCost

    x_test = testSet
    x_test = x_test.drop('tube_assembly_id', axis = 1)
    x_test = x_test.drop('id', axis=1)

    # Change the parameters here for experimentation
    rf = RandomForestRegressor(n_estimators=100, max_features=0.5, n_jobs=-1)
    rf = rf.fit(x_train, y_train)

    # Train RFR model
    trainPred = rf.predict(x_train)
    rmsle = mean_squared_error(y_train, trainPred)
    print "RMSLE Train: ",rmsle

    # Save train predictions to file
    traindf = pd.DataFrame()
    traindf['cost'] = inverseLogTransform(trainPred)
    traindf['id'] = traindf.index + 1
    file = open(os.path.join(RFRDir,'train_predictions_RFR.csv'),'w')
    traindf.to_csv(file, index=False, columns=['id','cost'])

    # Save Test predictions to file
    testdf = pd.DataFrame()
    testdf['cost'] = inverseLogTransform(rf.predict(x_test))
    testdf['id'] = testdf.index + 1
    file = open(os.path.join(RFRDir,'test_predictions_RFR.csv'),'w')
    testdf.to_csv(file, index=False, columns=['id','cost'])