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
    return rawDfs


def loadComponentData():
    componentTypes = pd.read_csv(os.path.join(dataDir,'type_component.csv'))
    componentGroups = ['adaptor','boss','elbow','float','hfl','nut','other','sleeve','straight','tee','threaded']

    #  Dictionary of dataframes for different component groups
    componentGroupsData = {}
    for component in componentGroups:
        componentGroupsData[component] = pd.read_csv(os.path.join(dataDir, 'comp_' + component + '.csv'))

    return componentTypes, componentGroupsData


def mergeTubeFeatures(raw, tubeDf, specsDf, bomDf, tubeEndDf):
    df = pd.merge(raw, tubeDf, on='tube_assembly_id')
    df = pd.merge(df, specsDf, on='tube_assembly_id')
    df = pd.merge(df, bomDf, on='tube_assembly_id')
    df = pd.merge(df, tubeEndDf, left_on="end_a", right_on="end_form_id", how='left')
    df = df.rename(columns={"forming": "end_a_forming"})
    df = df.drop("end_form_id", axis=1)
    df = pd.merge(df, tubeEndDf, left_on="end_x", right_on="end_form_id", how='left')
    df = df.rename(columns={'forming': 'end_x_forming'})
    df = df.drop("end_form_id", axis=1)
    # dfToCSV(df, 'merged_tube_features')
    return df


def getComponentFeatures(componentGroupsData):
    componentFeatures = []
    for component_df in componentGroupsData.itervalues():
        for col in component_df.columns:
            if(col not in componentFeatures):
                componentFeatures.append(col)

    # print "Component Features: ", componentFeatures
    # print "Number of component features: ", len(componentFeatures)
    return componentFeatures

def getBracketPricePatterns(df):
    grouped = df.groupby(
        ['tube_assembly_id', 'supplier', 'quote_date'])
    bracketing_pattern = [None] * len(df)
    for t_s_q, indices in grouped.groups.iteritems():
        if len(indices) > 1:
            bracket = tuple(sorted(df.adjusted_quantity[indices].values))
        else:
            bracket = ()
        for index in indices:
            bracketing_pattern[index] = bracket

    return bracketing_pattern


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

    aggregateCols = [
        ('component_length', 'component_length', max, 0.0),
        ('component_thread_pitch', 'component_thread_pitch', min, 9999),
        ('component_thread_size', 'component_thread_size', min, 9999),
    ]
    for new_col, col, aggregator, init in aggregateCols:
        mergedCompsDf[new_col] = mergedCompsDf[col].map(lambda x: aggregator(init, init, *x))

    colToReplaceNulls = {
        'component_type_id': 'other',
        'bolt_pattern_long': 0.0,
        'bolt_pattern_wide': 0.0,
        'overall_length': 0.0,
        'thickness': 0.0,
        'weight': 0.0,
    }

    for col, nullVal in colToReplaceNulls.iteritems():
        mergedCompsDf[col].fillna(nullVal, inplace=True)

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


def categoricalToNumeric(df, col_name, multiple = False, min_seen_count = 10, extractFeatures=True, sourceDf=None):
    counter = None
    val_to_int = None
    int_to_val = None

    # Build a map from string feature values to unique integers.
    # Assumes 'other' does not occur as a value.
    val_to_int = {'XXX_other': 0}
    int_to_val = ['XXX_other']
    next_index = 1
    counter = Counter()

    if(extractFeatures):
        extractable = df
    else:
        extractable = sourceDf

    for val in extractable[col_name]:
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
    # df = df.drop(col_name, axis=1)
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
    # dfToCSV(specDf,'tube_specs_as_list')
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
    # dfToCSV(componentsDf,'tube_components_as_list')
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

    # dfToCSV(pd.concat([df, featureDf], axis=1), 'train_set_after_merged_components')
    return pd.concat([df, featureDf], axis=1)


def getAdjustedQuantiy(df):
    return df[['min_order_quantity', 'quantity']].max(axis=1)

def getId(vals):
    return list(vals)

def getFlattenedList(lists):
    return list(itertools.chain(*lists))

def getSum(vals):
    return sum(vals)

def dropNulls(vals):
    return filter(lambda x: not pd.isnull(x), vals)

def yesNotoBinary(df, feature):
    for row in range(0,len(df)):
        if str.lower(str(df.get_value(row,feature))) in {'yes','y'}:
            df.set_value(row,feature,1)
        elif str.lower(str(df.get_value(row,feature))) in {'no','n','nan'}:
            df.set_value(row,feature,0)

    pd.to_numeric(df[feature])
    return df

def logTransform(costList):
    return np.log(costList + 1)

def inverseLogTransform(logCostList):
    return np.exp(logCostList) - 1

def mergeComponentFeatures(curDf, mergedComponents):

    mergedComponents = yesNotoBinary(mergedComponents, 'orientation')
    mergedComponents = yesNotoBinary(mergedComponents, 'unique_feature')
    mergedComponents = yesNotoBinary(mergedComponents, 'groove')
    dfToCSV(mergedComponents, 'merged_components')
    # Add features from the component_info_df.
    aggregatorCols = [
        ('component_groups', 'component_group_id', getId),
        ('unique_feature_count', 'unique_feature', getSum),
        ('component_types', 'component_type_id', getId),
        ('orientation_count', 'orientation', getSum),
        ('groove_count', 'groove', getSum),
        ('total_component_weight', 'weight', getSum),
        ('component_end_form', 'component_end_form', getFlattenedList),
        ('component_connection_type', 'component_connection_type', getFlattenedList),
        ('component_part_names', 'part_name', dropNulls),
    ]

    for feat, col, aggregator in aggregatorCols:
        cid_to_val = dict(zip(
            mergedComponents.component_id.values,
            mergedComponents[col].values))
        feat_vals = []
        for cid_list in curDf.components:
            vals = [cid_to_val[cid] for cid in cid_list]
            feat_vals.append(aggregator(vals))
        curDf[feat] = feat_vals

    aggregateColsNumericType = [
        ('component_max_length', 'component_length', max, 0.0),
        ('component_max_overall_length', 'overall_length', max, 0.0),
        ('component_max_bolt_pattern_wide', 'bolt_pattern_wide', max, 0.0),
        ('component_max_bolt_pattern_long', 'bolt_pattern_long', max, 0.0),
        ('component_max_thickness', 'thickness', max, 0.0),
        ('component_min_thread_pitch', 'component_thread_pitch', min, 9999),
        ('component_min_thread_size', 'component_thread_size', min, 9999),
    ]


    for feat, col, aggregator, init in aggregateColsNumericType:
        cid_to_val = dict(zip(
            mergedComponents.component_id.values,
            mergedComponents[col].values))
        feat_vals = []
        for cid_list in curDf.components:
            vals = [cid_to_val[cid] for cid in cid_list]
            feat_vals.append(aggregator(init, init, *vals))
        curDf[feat] = feat_vals

    return curDf

# Get 3 features: end_forming_count, end_1x_count, end_2x_count
# end_forming_count = end_a_forming + end_x_forming
# end_1x_count = end_a_1x + end_x_1x
# end_2x_count = end_a_2x + end_x_2x
def getEndsFeatures(df):
    df['end_forming_count'] = df.end_a_forming.add(df.end_x_forming)
    df['end_1x_count'] = df['end_a_1x'].add(df['end_x_1x'])
    df['end_2x_count'] = df['end_a_2x'].add(df['end_x_2x'])
    return df

def getAugmentedDataset(raw, mergedComponents, specsDf, bomDf, tubeEndDf, tubeDf, extractFeatures=True, sourceDf=None):
    # Get specs feature with list of values for different specs
    specsDf = getSpecsAsList(specsDf)
    # Get component from bill_of_material as list of values under one feature components
    bomDf = getComponentsAsList(bomDf)
    # Merge tube features i.e. tube, bom, train/test, spec, endform
    augDf = mergeTubeFeatures(raw, tubeDf, specsDf, bomDf, tubeEndDf)

    # Convert binary feature (Yes No values ones) to 0 or 1
    features = ['bracket_pricing', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a_forming', 'end_x_forming']
    for feat in features:
        augDf = yesNotoBinary(augDf, feat)

    # Composite features
    augDf['quote_age'] = getQuoteAge(augDf)                           # Quote_age
    augDf['adjusted_quantity'] = getAdjustedQuantiy(augDf)            # Adjusted quantity
    augDf = getPhysicalMaterialVolume(augDf)                          # Physical and material volume
    augDf['bracket_price_pattern'] = pd.Series(getBracketPricePatterns(augDf))
    # augDf = getEndsFeatures(augDf)

    # Merge component features with dataset
    augDf = mergeComponentFeatures(augDf, mergedComponents)

    # Drop tube_assembly_id and quote_date
    # augDf = augDf.drop('tube_assembly_id', axis=1)
    augDf = augDf.drop('quote_date', axis=1)

    # Convert categorical to numerical
    categories = [
        ('supplier', False),
        ('specs', True),
        ('material_id', False ),
        ('end_a', False),
        ('end_x', False),
        ('components', True),
        ('bracket_price_pattern', False),
        ('component_groups', True),
        ('component_types', True),
        ('component_end_form', True),
        ('component_connection_type', True),
        ('component_part_names', True),
    ]

    for colName, multiBool in categories:
        augDf = categoricalToNumeric(augDf, colName, multiple=multiBool, min_seen_count=30, extractFeatures=extractFeatures, sourceDf=sourceDf)

    return augDf


def getFinalTrainAndTestSet():
    raw = loadRawData()
    tubeDf = raw['tube']
    tubeEndDf = raw['tube_end_form']
    specsDf = raw['specs']
    bomDf = raw['bill_of_materials']
    mergedComponents = mergeComponents()
    trainSet = getAugmentedDataset(raw['train_set'], mergedComponents, specsDf, bomDf, tubeEndDf, tubeDf, extractFeatures=True, sourceDf=None)
    trainSet['log_cost'] = logTransform(trainSet.pop('cost'))
    # dfToCSV(trainSet, 'train_set_merged')
    testSet = getAugmentedDataset(raw['test_set'], mergedComponents, specsDf, bomDf, tubeEndDf, tubeDf, extractFeatures=False, sourceDf=trainSet)

    # Convert categorical to numerical
    categories = ['supplier', 'specs', 'material_id', 'end_a', 'end_x', 'components', 'bracket_price_pattern', 'component_groups', 'component_types', 'component_end_form', 'component_connection_type', 'component_part_names']

    for category in categories:
        trainSet = trainSet.drop(category, axis=1)
        testSet = testSet.drop(category, axis=1)

    dfToCSV(trainSet, 'train_set_merged')
    dfToCSV(testSet, 'test_set_merged')

    print "List of trainset: ", trainSet.columns
    print "List of testset: ", testSet.columns
    print "Num of train cols: ", len(trainSet.columns)
    print "Num of test cols: ", len(testSet.columns)
    return trainSet, testSet

# getFinalTrainAndTestSet()