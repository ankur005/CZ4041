from datetime import datetime
from collections import Counter
import pandas as pd
import os
from utility import outDir, dataDir

def loadRawData():
    # load main files in dataframe
    fileNames = ['train_set', 'test_set', 'tube', 'specs', 'bill_of_materials', 'tube_end_form']
    # Dictionary of dataframes loaded from the raw data files as provided in the competition data
    rawDfs = {}
    for file in fileNames:
        rawDfs[file] = pd.read_csv(os.path.join(dataDir, file + '.csv'))

    #
    # # Process Specs Data
    # specsDf = pd.DataFrame()
    # specsDf['tube_assembly_id'] = rawDfs['specs']['tube_assembly_id']
    # tmp_df = rawDfs['specs'].where(pd.notnull(rawDfs['specs']), None)
    # specs = [filter(None, row[1:]) for row in tmp_df.values]
    # specsDf['specs'] = specs
    # file = open(os.path.join(outDir,'new_specs.csv'),'w')
    # specsDf.to_csv(file)
    # rawDfs['specs'] = specsDf
    #
    # # Process Components data
    # componentsDf = pd.DataFrame()
    # componentsDf['tube_assembly_id'] = rawDfs['bill_of_materials']['tube_assembly_id']
    # tmp_df = rawDfs['bill_of_materials'].where(pd.notnull(rawDfs['bill_of_materials']), None)
    # components = []
    # for origRow in (filter(None, row[1:]) for row in tmp_df.values):
    #     newRow = []
    #     for component_str, count in zip(origRow[0::2], origRow[1::2]):
    #         assert int(count) == count
    #         newRow.extend([component_str] * int(count))
    #     components.append(newRow)
    # componentsDf['components'] = components
    # rawDfs['components'] = componentsDf
    # file = open(os.path.join(outDir, 'new_components.csv'), 'w')
    # componentsDf.to_csv(file)
    return rawDfs


def loadComponentData():
    componentTypes = pd.read_csv(os.path.join(dataDir,'type_component.csv'))
    componentGroups = ['adaptor','boss','elbow','float','hfl','nut','other','sleeve','straight','tee','threaded']

    #  Dictionary of dataframes for different component groups
    componentGroupsData = {}
    for component in componentGroups:
        componentGroupsData[component] = pd.read_csv(os.path.join(dataDir, 'comp_' + component + '.csv'))

    return componentTypes, componentGroupsData


def mergeTubeFeatures(raw):
    trainDataDf = pd.merge(raw['train_set'], raw['tube'], on='tube_assembly_id')
    trainDataDf = pd.merge(trainDataDf, raw['specs'], on='tube_assembly_id')
    trainDataDf = pd.merge(trainDataDf, raw['bill_of_materials'], on='tube_assembly_id')
    trainDataDf = pd.merge(trainDataDf, raw['tube_end_form'], left_on="end_a", right_on="end_form_id", how='left')
    trainDataDf = trainDataDf.rename(columns={"forming": "end_a_forming"})
    trainDataDf = trainDataDf.drop("end_form_id", axis=1)
    trainDataDf = pd.merge(trainDataDf, raw['tube_end_form'], left_on="end_x", right_on="end_form_id", how='left')
    trainDataDf = trainDataDf.rename(columns={'forming': 'end_x_forming'})
    trainDataDf = trainDataDf.drop("end_form_id", axis=1)

    dfToCSV(trainDataDf, 'merged_tube_features')


def getComponentFeatures(componentGroupsData):
    componentFeatures = []
    for component_df in componentGroupsData.itervalues():
        for col in component_df.columns:
            if(col not in componentFeatures):
                componentFeatures.append(col)

    print "Component Features: ", componentFeatures
    print "Number of component features: ", len(componentFeatures)
    return componentFeatures


def mergeComponents():
    componentTypes, componentGroupsData = loadComponentData()
    # componentFeatures = getComponentFeatures(componentGroupsData)
    compDfList = []

    for (component, componentDf) in componentGroupsData.iteritems():
        componentDf['component_group_id'] =  component
        compDfList.append(componentDf)

    mergedCompsDf = pd.concat(compDfList, axis=0, ignore_index=True)
    dfToCSV(mergedCompsDf, 'merged_comps')
    return mergedCompsDf


def dfToCSV(df, fileName):
    with open(os.path.join(outDir,fileName + '.csv'),'wb') as file:
        df.to_csv(file)


def oneHotEncoder(dataframe, column):
    dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[column])], axis=1)
    return dataframe


def getAugmentedDataset(tubeDf, mergedComponents):
    # One hot encode supplier labels
    tubeDf = oneHotEncoder(tubeDf, 'supplier')
    # One hot encode material ID labels
    tubeDf = oneHotEncoder(tubeDf, 'material_id')
    dfToCSV(tubeDf, 'merged_tube_features')


raw = loadRawData()
tubeDf = mergeTubeFeatures(raw)
mergedComponents = mergeComponents()
getAugmentedDataset(tubeDf, mergedComponents)