# created by Noel Konagai at 2019/11/22 13:37.
# 
# This code was written by Noel Konagai.

import pandas as pd
import numpy as np
import os, glob

from sklearn import preprocessing 

def get_filenames(path, extension):
    os.chdir(path)
    filenames = [f for f in glob.glob(extension)]
    os.chdir("..")
    return filenames

def concat_df():
    '''
    This function gets all the dataframes in in_dir and creates
    a master dataframe with the file names as keys in out_dir.
    '''
    filenames = get_filenames(in_dir, "*.csv")
    keys = []

    frames = []

    for filename in filenames:
        df = pd.read_csv(in_dir + filename, index_col = 0)
        frames.append(df)
        keys.append(filename[:-11])

    result = pd.concat(frames, keys = keys)

    return result 

def create_author_id(df):
    '''
    In order to investigate whether any message authors are part of other groups,
    this function creates a label encoding of authors in author_id column.
    '''
    le = preprocessing.LabelEncoder()
    df = df.drop(['sender_id'], axis=1)
    fit = le.fit_transform(df['author'])
    df['author_id'] = fit
    np.save(out_dir + 'author_id_classes.npy', le.classes_)
    return df

def categorize_groups(df):
    '''
    This function categorizes the groups into 1, 2, 3 depending on whether they were
    government teacher, meghshala, or 321 groups.
    '''
    gov = ['subramanya', 'smart_hms', 'suliya', 'nallikalli', 'hindalga', 'marihal', 'chikabagawadi', 'belgavi']
    meghshala = ['meghshala-balpa', 'meghshala-belgaum-1', 'meghshala-belgaum-2', 'meghshala-kundagol', 'meghshala-msouth-1', 'meghshala-msouth-2', 'meghshala-mysore'] #this is not all I think
    tto = ['rosemary', 'pioneer', 'msn', 'morningstar', 'maideal', 'kwy1', 'ju', 'holystar', 'gla', 'ekids', 'biy1']

    group_type = []

    for i in range(len(df)):
        if df.iloc[i].name in gov:
            group_type.append(1)
        elif df.iloc[i].name in meghshala:
            group_type.append(2)
        elif df.iloc[i].name in tto:
            group_type.append(3)
        else:
            print("Filename not listed for:", df.iloc[i].name)

    df.insert(1, 'group_type', group_type)

    return df
    
if __name__ == "__main__":
    #TODO Read one by one chkpt2 CSV files
    #TODO Recursively call create_master_df

    in_dir = "../data/chkpt2/"
    out_dir = "../data/chkpt3/"

    result = concat_df()
    df = create_author_id(result)   
    df.to_csv(out_dir + "master.csv", index_label = "index")
    df = pd.read_csv(out_dir + "master.csv")
    df = categorize_groups(df)
    df.to_csv(out_dir + "master.csv")
