from datetime import datetime
from collections import Counter

import itertools
import pandas as pd
import os
import math
import numpy as np
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
    return trainDataDf


def getComponentFeatures(componentGroupsData):
    componentFeatures = []
    for component_df in componentGroupsData.itervalues():
        for col in component_df.columns:
            if(col not in componentFeatures):
                componentFeatures.append(col)

    print "Component Features: ", componentFeatures
    print "Number of component features: ", len(componentFeatures)
    return componentFeatures

def getBracketPricePatterns(df):
    grouped = df.groupby(
        ['tube_assembly_id', 'supplier', 'quote_date'])
    bracketing_pattern = [None] * len(df)
    for t_s_q, indices in grouped.groups.iteritems():
        if len(indices) > 1:
            bracket = tuple(sorted(df.quantity[indices].values))
        else:
            bracket = ()
        for index in indices:
            bracketing_pattern[index] = bracket

    print bracketing_pattern
    return bracketing_pattern

    # bracketedRecords = [filter(lambda i: df.get_value(i,'bracket_pricing') == "Yes", df) for i in range(0, len(df))]
    # bracketedRecordsDf = pd.DataFrame.from_records(bracketedRecords)
    # uniqueTaids = bracketedRecordsDf['tube_assembly_id'].unique()
    #
    # bracketPatterns = []
    #
    # for taid in uniqueTaids:
    #     bracketPattern = []
    #     taidDf = bracketedRecordsDf.query('tube_assembly_id == taid')
    #     for i in range(0, len(taidDf)):
    #         bracketPattern.append(taidDf.get_value(i,'quantity'))
    #
    #     bracketPattern = tuple(bracketPattern)
    #
    # if bracketPattern not in bracketPatterns:
    #     bracketPatterns.append(bracketPattern)
    #
    # print "Bracket Patterns: ", bracketPatterns


def mergeComponents():
    componentTypes, componentGroupsData = loadComponentData()
    # componentFeatures = getComponentFeatures(componentGroupsData)
    compDfList = []

    for (component, componentDf) in componentGroupsData.iteritems():
        componentDf['component_group_id'] =  component
        compDfList.append(componentDf)

    mergedCompsDf = pd.concat(compDfList, axis=0, ignore_index=True)
    dfToCSV(mergedCompsDf, 'merged_comps_initial')

    combineColsDict = {'component_length' : ['length', 'length_1', 'length_2', 'length_3', 'length_4' ],
                   'component_end_form' : ['end_form_id_1', 'end_form_id_2', 'end_form_id_3', 'end_form_id_4'],
                   'component_connection_type' : ['connection_type_id', 'connection_type_id_1', 'connection_type_id_2', 'connection_type_id_3', 'connection_type_id_4'],
                   'component_nominal_size' : ['nominal_size_1', 'nominal_size_2', 'nominal_size_3', 'nominal_size_4'],
                   'component_thread_size': ['thread_size', 'thread_size_1', 'thread_size_2', 'thread_size_3', 'thread_size_4'],
                   'component_thread_pitch' : ['thread_pitch', 'thread_pitch_1', 'thread_pitch_2', 'thread_pitch_3', 'thread_pitch_4']}

    for key in combineColsDict:
        df = mergedCompsDf[combineColsDict[key]]
        newDf = df.where(pd.notnull(df), None)
        colList = [filter(None, row[0:]) for row in newDf.values]
        mergedCompsDf[key] =  colList
        mergedCompsDf = mergedCompsDf.drop(combineColsDict[key], axis=1)

    dfToCSV(mergedCompsDf, 'merged_components')
    return mergedCompsDf


def dfToCSV(df, fileName):
    with open(os.path.join(outDir,fileName + '.csv'),'wb') as file:
        df.to_csv(file)


def oneHotEncoder(df, column, dummy, prefix):
    # print pd.get_dummies(df[column])
    return pd.concat([df, pd.get_dummies(df[column], dummy_na=dummy, prefix=prefix)], axis=1)


def getQuoteAge(df):
    series = pd.to_datetime(df['quote_date']) - datetime(1900, 1, 1)
    return series.astype('timedelta64[D]')

def categoricalToNumeric(df, col_name, multiple = False, min_seen_count = 10):
    counter = None
    val_to_int = None
    int_to_val = None

    # Build a map from string feature values to unique integers.
    # Assumes 'other' does not occur as a value.
    val_to_int = {'XXX_other': 0}
    int_to_val = ['XXX_other']
    next_index = 1
    counter = Counter()
    for val in df[col_name]:
        if multiple:
            # val is a list of categorical values.
            counter.update(val)
        else:
            # val is a single categorical value.
            counter[val] += 1
    for val, count in counter.iteritems():
        if count >= min_seen_count:
            val_to_int[val] = next_index
            int_to_val.append(val)
            next_index += 1

    feats = np.zeros((len(df), len(val_to_int)))
    for i, orig_val in enumerate(df[col_name]):
        if multiple:
            # orig_val is a list of categorical values.
            list_of_vals = orig_val
        else:
            # orig_val is a single categorical value.
            list_of_vals = [orig_val]
        for val in list_of_vals:
            if val in val_to_int:
                feats[i, val_to_int[val]] += 1
            else:
                feats[i, val_to_int['XXX_other']] += 1
    feat_names = ['{} {}'.format(col_name, val) for val in int_to_val]
    df = df.drop(col_name, axis=1)
    return pd.concat([df, pd.DataFrame(feats, index=df.index, columns=feat_names)], axis=1)


def getSpecsAsList(df):
    specDf = pd.DataFrame()
    specDf['tube_assembly_id'] = df['tube_assembly_id']
    newDf = df.where(pd.notnull(df), None)
    specsList = [filter(None, row[1:]) for row in newDf.values]
    # for i in range(0, len(df)):
    #     tempList = []
    #     for j in range(1,11):
    #         tempList.append(df.get_value(i, 'spec' + str(j)))
    #     specsList.append([x for x in tempList if not(pd.isnull(x))])
    #     newDf.set_value(i, 'spec', tempList)
    # print specsList
    specDf['specs'] = specsList
    dfToCSV(specDf,'tube_specs_as_list')
    return specDf

def getComponentsAsList(df):
    componentsDf = pd.DataFrame()
    componentsDf['tube_assembly_id'] = df['tube_assembly_id']
    #  Sets the value of the cells which have NA value to empty
    newDf = df.where(pd.notnull(df), None)

    components = []
    # Original row consists of the column: component_id_[1:8] and quantity_[1:8]
    for origRow in (filter(None, row[1:]) for row in newDf.values):
        # In new row, quantity will be removed and component ID will be replicated for count greater than 1
        newRow = []
        for component_str, count in zip(origRow[0::2], origRow[1::2]):
            assert int(count) == count
            newRow.extend([component_str] * int(count))
        components.append(newRow)
    componentsDf['components'] = components
    dfToCSV(componentsDf,'tube_components_as_list')
    return componentsDf

def getPhysicalMaterialVolume(df):
    # Can explore effects of inner radius or inner volume
    df['physical_volume'] = 0.0
    df['material_volume'] = 0.0
    for i in range(0, len(df)):
        radius = df.get_value(i, 'diameter') / 2
        length = df.get_value(i, 'length')
        thickness = df.get_value(i, 'wall')
        phyVolume = (math.pi * (radius**2) * length)
        innerVolume = (math.pi * ((radius-thickness)**2) * length)
        df.set_value(i, 'physical_volume', phyVolume)
        df.set_value(i, 'material_volume', innerVolume)

    dfToCSV(df[['tube_assembly_id','physical_volume','material_volume']],'volume_tubes')
    return df


def componentToFeatures(df):
    featureDf = pd.DataFrame()
    for i in range(0,len(df)):
        print i
        for j in range(1,9):
            componentId = df.get_value(i, 'component_id_' + str(j))
            quantity = df.get_value(i, 'quantity_' + str(j))
            if pd.isnull(quantity):
                quantity = 0
            if not(pd.isnull(componentId)):
                print "Here: ", componentId, quantity
                colName = 'component_id_' + str(componentId)
                if colName not in featureDf.columns:
                    featureDf[colName] = 0
                featureDf.set_value(i, colName, quantity)

    return pd.concat([df, featureDf], axis=1)
    dfToCSV(pd.concat([df, featureDf], axis=1), 'train_set_after_merged_components')

def myidentity(init0, init1, *vals):
    return list(vals)

def flatten(*lists):
    return list(itertools.chain(*lists))

def mysum(*vals):
    sum(vals)

def mergeTrainAndComponentFeatures(curTraindf, mergedComponents):
    # Add features from the component_info_df.
    aggregations = [
        ('unique_feature_count', 'unique_feature', mysum, 0.0)
    ]
    for feat, col, aggregator, init in aggregations:
        cid_to_val = dict(zip(
            mergedComponents.component_id.values,
            mergedComponents[col].values))
        print cid_to_val
        feat_vals = []
        for cid_list in curTraindf.components:
            vals = [cid_to_val[cid] for cid in cid_list]
            feat_vals.append(aggregator(*vals))
        print feat_vals
        curTraindf[feat] = feat_vals
        return curTraindf


def getAugmentedDataset(raw, mergedComponents):
    # Get specs feature with list of values for different specs
    raw['specs'] = getSpecsAsList(raw['specs'])
    # Get component from bill_of_material as list of values under one feature components
    raw['bill_of_materials'] = getComponentsAsList(raw['bill_of_materials'])
    tubeDf = mergeTubeFeatures(raw)
    # One hot encode supplier labels
    tubeDf = oneHotEncoder(tubeDf, 'supplier', True, 'supplier')
    # One hot encode material ID labels
    tubeDf = oneHotEncoder(tubeDf, 'material_id', True, 'material')
    # One hot encode end_a column
    tubeDf = oneHotEncoder(tubeDf, 'end_a', True, 'end_a')
    # One hot encode end_x column
    tubeDf = oneHotEncoder(tubeDf, 'end_x', True, 'end_x')

    # Convert to Numeric components column
    tubeDf = categoricalToNumeric(tubeDf, 'components', multiple=True, min_seen_count=30)
    # Convert to Numeric specs column
    tubeDf = categoricalToNumeric(tubeDf, 'specs', multiple=True, min_seen_count=30)

    bracketPatterns = getBracketPricePatterns(tubeDf)
    tubeDf['bracket_price_pattern'] = pd.Series(bracketPatterns)
    dfToCSV(tubeDf, 'train_set_merged')

    # Quote age feature
    tubeDf['quote_age'] = getQuoteAge(tubeDf)
    # Physical and material volume feature
    tubeDf = getPhysicalMaterialVolume(tubeDf)
    # Merge component features with dataset
    tubeDf = mergeTrainAndComponentFeatures(tubeDf, mergedComponents)
    dfToCSV(tubeDf, 'train_set_merged')


raw = loadRawData()
mergedComponents = mergeComponents()
getAugmentedDataset(raw, mergedComponents)
