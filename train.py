import xgboost as xgb
import datetime
import numpy as np
import os
import pandas as pd
from generateDataSets import getFinalTrainAndTestSet, inverseLogTransform, logTransform
from sklearn.metrics import mean_squared_error


xgbParams = {
    'objective': 'reg:linear',
    'silent': 1,
    'num_rounds': 10000,
    'gamma': 0.0,
    'eta': 0.02,
    'max_depth': 8,
    'min_child_weight': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.6
}

#  Generate bags - Feature Set 2 - Feature Set 1 for k(10)-fold cross validation
def generateBags(trainSet):
    # print "Received Train Set Features : ", list(trainSet)
    uniqueTubeAssemblyIds = np.unique(trainSet.tube_assembly_id.values)
    # print "Unique ", uniqueTubeAssemblyIds
    print "Total rows : ", len(trainSet), "Unique Tube assembly Ids : ", len(uniqueTubeAssemblyIds)
    numUniqueIdsBag = 0.9 * len(uniqueTubeAssemblyIds)
    # print "Unique IDs in Bag :", numUniqueIdsBag
    random = np.random.RandomState()

    bagTubeIds = random.choice(uniqueTubeAssemblyIds, size=int(numUniqueIdsBag), replace=False)
    # print "Bag Tube Ids", bagTubeIds
    # uniqueBagIds = np.unique(bagTubeIds)
    bag = trainSet.tube_assembly_id.isin(bagTubeIds)
    # print "Bag", list(bag.index)
    trainSet = trainSet[bag].reset_index(drop=True)
    # print "Train Set", trainSet
    return trainSet


def trainModel(trainSet, path):
    # print "Rec Train Set ", list(trainSet)
    trainSet = trainSet.drop('tube_assembly_id', axis=1)
    # print "Train Set Proc ", list(trainSet)

    y_train = trainSet.log_cost.values
    x_train = trainSet.drop('log_cost', axis=1)

    # print "X Train ", list(x_train)
    # print "X Train Shape ", x_train.shape
    # print "Y Train Shape ", y_train.shape
    x_train = np.array(x_train).astype(float)
    # print "xtrain np array  ", x_train.shape

    xgtrain = xgb.DMatrix(x_train, label=y_train)
    model = xgb.train(list(xgbParams.items()), xgtrain, xgbParams['num_rounds'])
    os.mkdir(path)
    model.save_model(os.path.join(path,'model'))


def predict(dataset, path):
    dataset = dataset.drop('tube_assembly_id', axis=1)
    dataset = np.array(dataset).astype(float)
    predictions = None
    model = xgb.Booster()
    model.load_model(os.path.join(path,'model'))

    xgeval = xgb.DMatrix(dataset)
    predictions = model.predict(xgeval)
    df = pd.DataFrame()
    df['cost'] = predictions
    # print df['cost']
    df['id'] = df.index + 1
    return df



if __name__ == '__main__':
    print "AT: ", datetime.datetime.now(),
    print "Load datasets..."
    # trainSet, testSet = getFinalTrainAndTestSet()
    trainSet = pd.read_csv('./OutData/train_set_merged.csv')
    testSet = pd.read_csv('./OutData/test_set_merged.csv')
    testSet = testSet.drop('id', axis=1)
    # print "Train Features : ", list(trainSet)
    # print "\n\n......."
    # print "Test Features : ", list(testSet)
    # print trainSet.shape
    # print testSet.shape
    dir = os.path.dirname(os.path.join(os.getcwd(), "bags - Feature Set 4"))
    dirPath = os.path.join(os.getcwd(), "bags - Feature Set 4")
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    for i in range(0, 10):
        path = os.path.join(dirPath, str(i))
        print "AT: ", datetime.datetime.now(),
        print "TRAINING............................."
        print "Selecting bag ", i, "..."
        trainBag = generateBags(trainSet)
        print "Training bag ", i
        trainModel(trainBag, path)
        print "Predicting for Train Data...."
        trainCost = trainBag.pop('log_cost')
        trainPredictions = predict(trainBag, path)
        trainRMSLE = np.sqrt(mean_squared_error(trainCost.values, trainPredictions['cost'].values))
        trainPredictions['cost'] = inverseLogTransform(trainPredictions['cost'])        # Inverse log transform the predictions before saving to file
        trainPredFile = open(os.path.join(path, 'train_predictions.csv'), 'w')
        trainPredictions.to_csv(trainPredFile, index=False, columns=['id','cost'])
        trainPredFile.close()
        print "RMSLE: ", trainRMSLE
        print "AT: ", datetime.datetime.now(),
        print "Predicting for Test Data...."
        testPredictions = predict(testSet, path)
        testPredictions['cost'] = inverseLogTransform(testPredictions['cost'])          # Inverse log transform the predictions before saving to file
        testPredFile = open(os.path.join(path, 'test_predictions.csv'), 'w')
        testPredictions.to_csv(testPredFile, index=False, columns=['id','cost'])
        testPredFile.close()
        print "AT: ", datetime.datetime.now(),
        print "Bag ", i, "Done!"
    print "AT: ", datetime.datetime.now(),
    print "IT'S OVER!"
