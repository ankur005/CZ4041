from datetime import datetime
from collections import Counter
import pandas as pd
import os

dataDir = "./Data"
outDir = "./OutData"

def loadRawData():
    fileNames = ['train_set', 'test_set', 'tube', 'specs', 'bill_of_materials', 'tube_end_form']
    # Dictionary of dataframes loaded from the raw data files as provided in the competition data
    rawDfs = {}
    for file in fileNames:
        rawDfs[file] = pd.read_csv(os.path.join(dataDir, file + '.csv'))

    # Process Specs Data
    specsDf = pd.DataFrame()
    specsDf['tube_assembly_id'] = rawDfs['specs']['tube_assembly_id']
    tmp_df = rawDfs['specs'].where(pd.notnull(rawDfs['specs']), None)
    specs = [filter(None, row[1:]) for row in tmp_df.values]
    specsDf['specs'] = specs

    file = open(os.path.join(outDir,'.new_specs.csv'),'w')
    specsDf.to_csv(file)
    rawDfs['specs'] = specsDf
    return rawDfs

def loadComponentData():
    componentTypes = pd.read_csv(os.path.join(dataDir,'type_component.csv'))
    componentGroups = ['adaptor','boss','elbow','float','hfl','nut','other','sleeve','straight','tee','threaded']

    #  Dictionary of dataframes for different component groups
    componentGroupsData = {}
    for component in componentGroups:
        componentGroupsData[component] = pd.read_csv(os.path.join(dataDir, 'comp_' + component + '.csv'))

    return componentTypes, componentGroupsData

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
    componentFeatures = getComponentFeatures(componentGroupsData)
    comp_df_list = []

    for (component, component_df) in componentGroupsData.iteritems():
        component_df['component_group_id'] =  component
        comp_df_list.append(component_df)

    merged_comps_df = pd.concat(comp_df_list, axis=0, ignore_index=True)

    with open(os.path.join(outDir,'merged_comps.csv'),'wb') as fobj:
        merged_comps_df.to_csv(fobj)
    return merged_comps_df

# raw = loadRawData()
# componentTypes, componentGroupsData = loadComponentData()
# componentFeatures = getComponentFeatures(componentGroupsData)
mergedComponents = mergeComponents()