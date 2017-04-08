import xgboost as xgb
import pickle
import numpy as np
import os
import pandas as pd
from generateDataSets import getFinalTrainAndTestSet, inverseLogTransform
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

#  Generate bags for k(10)-fold cross validation
def generateBags(trainSet):
    uniqueTubeAssemblyIds = np.unique(trainSet.tube_assembly_id.values)
    print "Total rows : ", len(trainSet), "Unique Tube assembly Ids : ", len(uniqueTubeAssemblyIds)
    numUniqueIdsBag = 0.9 * len(uniqueTubeAssemblyIds)
    random = np.random.RandomState()

    bagTubeIds = random.choice(uniqueTubeAssemblyIds, size=int(numUniqueIdsBag), replace=False)
    # uniqueBagIds = np.unique(bagTubeIds)
    bag = trainSet.tube_assembly_id.isin(bagTubeIds)
    trainSet = trainSet[bag].reset_index(drop=True)
    return trainSet


def trainModel(trainSet, testSet, path):
    trainSet = trainSet.drop('tube_assembly_id', axis=1)

    y_train = trainSet.log_cost.values
    x_train = np.array(trainSet).astype(float)

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

    predictions = model.predict(dataset)
    df = pd.DataFrame()
    df['cost'] = inverseLogTransform(predictions)
    df['id'] = df.index + 1
    return df


if __name__ == '__main__':
    print "Load datasets..."
    trainSet, testSet = getFinalTrainAndTestSet()
    # trainSet = pd.read_csv('./OutData/train_set_merged.csv')
    # testSet = pd.read_csv('./OutData/test_set_merged.csv')
    # dir = os.path.dirname(os.path.join(os.getcwd(), "bags"))
    dirPath = os.path.join(os.getcwd(), "bags")
    os.mkdir(dirPath)
    for i in range(0,10):
        path = os.path.join(dirPath, str(i))
        print "TRAINING............................."
        print "Selecting bag ", i, "..."
        trainBag = generateBags(trainSet)
        print "Training bag ", i
        trainModel(trainBag, testSet, path)
        print "Predicting for Train Data...."
        trainCost = trainBag.pop('log_cost')
        trainPredictions = predict(trainBag, path)
        trainPredFile = open(os.path.join(path, 'train_predictions.csv'), 'w')
        trainPredictions.to_csv(trainPredFile)
        trainRMSLE = np.sqrt(mean_squared_error(trainCost.values, trainPredictions))
        print "RMSLE: ", trainRMSLE
        print "Predicting for Test Data...."
        testPredictions = predict(testSet, path)
        testPredFile = open(os.path.join(path, 'test_predictions.csv'), 'w')
        testPredictions.to_csv(testPredFile)
        print "Bag ", i, "Done!"

    print "IT'S OVER!"


