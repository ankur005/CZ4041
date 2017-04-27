from generateDataSets import getFinalTrainAndTestSet, inverseLogTransform
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
import os
from constants import RegressionDir

# Get merged train and test dataset
def getDataset():
    print "Reading dataset from file ..."
    trainSet, testSet = getFinalTrainAndTestSet()
    # Modify this list to drop certain columns for experimentation
    dropcols = ['quantity','end_a_1x','end_a_2x','end_x_1x','end_x_2x','end_a_forming','end_x_forming']
    trainSet = trainSet.drop(dropcols, axis=1)
    testSet = testSet.drop(dropcols, axis=1)
    trainSet = trainSet.drop('tube_assembly_id', axis=1)
    trainLogCost = trainSet.pop('log_cost')
    x_train = trainSet
    y_train = trainLogCost

    x_test = testSet
    x_test = x_test.drop('tube_assembly_id', axis=1)
    x_test = x_test.drop('id', axis=1)

    return x_train.as_matrix(), y_train.as_matrix(), x_test.as_matrix()


# Iterates over different alpha values to find lasso model with minimum RMSLE on validation set
def findBestRegressor(x_train_matrix, y_train_matrix, numFolds):
    min_rmsle = -1
    selectedAlpha = -1
    for i in range(1, 6):
        alpha = 0.1 * (10**(-i))
        print "alpha: " + str(alpha)
        rmsle = crossValidate(x_train_matrix, y_train_matrix, numFolds, alpha)

        if(min_rmsle == -1 or rmsle < min_rmsle):
            min_rmsle = rmsle
            selectedAlpha = alpha

    regressor = trainLasso(x_train_matrix, y_train_matrix, selectedAlpha)

    print "selected alpha: " + str(selectedAlpha)

    return regressor, selectedAlpha


# Performs cross validation
def crossValidate(x_train_matrix, y_train_matrix, numFolds, alpha):
    best_regressor = Lasso(alpha=alpha, max_iter=1000, normalize=True)
    min_rmsle = -1
    foldNum = 1
    avgRMSLE = 0

    print "Splitting into " + str(numFolds) + " folds ..."
    kf = KFold(n_splits=numFolds)

    for trainIndex, valIndex in kf.split(x_train_matrix, y=y_train_matrix):
        print "Fold: ", foldNum
        x_train_fold, x_val_fold = x_train_matrix[trainIndex], x_train_matrix[valIndex]
        y_train_fold, y_val_fold = y_train_matrix[trainIndex], y_train_matrix[valIndex]

        regressor = trainLasso(x_train_fold, y_train_fold, alpha)

        valPred = regressor.predict(x_val_fold)
        rmsle = mean_squared_error(y_val_fold, valPred)
        avgRMSLE = avgRMSLE + rmsle
        print "RMSLE Val: ", rmsle

        if (min_rmsle == -1 or rmsle < min_rmsle):
            min_rmsle = rmsle
            best_regressor = regressor

        foldNum = foldNum + 1

    avgRMSLE = avgRMSLE/numFolds
    print "min_rmsle: ", min_rmsle
    return avgRMSLE


# Trains a lasso model for given dataset
def trainLasso(x_train, y_train, alpha):
    print "Training ..."
    # regressor = LinearRegressor(normalize=True, n_jobs=-1)
    regressor = Lasso(alpha=alpha, max_iter=1000, normalize=True)
    regressor = regressor.fit(x_train, y_train)
    print "Training completed ..."
    return regressor


# Main function to run
if __name__ == '__main__':
    x_train_matrix, y_train_matrix, x_test_matrix = getDataset()
    # Use this to run for  specific value of alpha without cross validation
    # regressor = trainLasso(x_train_matrix, y_train_matrix, 0.01)
    regressor, alpha = findBestRegressor(x_train_matrix, y_train_matrix, 10)

    trainPred = regressor.predict(x_train_matrix)
    rmsle = mean_squared_error(y_train_matrix, trainPred)
    print "RMSLE entire Train Set: ",rmsle

    testPred = regressor.predict(x_test_matrix)


    # Save train predictions to file
    print "Saving training preds to file ..."
    traindf = pd.DataFrame()
    traindf['cost'] = inverseLogTransform(trainPred)
    traindf['id'] = traindf.index + 1
    file = open(os.path.join(RegressionDir,'train_predictions_lasso.csv'),'w')
    traindf.to_csv(file, index=False, columns=['id','cost'])

    # Save Test predictions to file
    print "Saving testing preds to file ..."
    testdf = pd.DataFrame()
    testdf['cost'] = inverseLogTransform(testPred)
    testdf['id'] = testdf.index + 1
    file = open(os.path.join(RegressionDir,'test_predictions_lasso.csv'),'w')
    testdf.to_csv(file, index=False, columns=['id','cost'])

    print "Done!"
