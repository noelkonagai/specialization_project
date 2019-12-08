# created by Noel Konagai at 2019/11/25 20:18.
# 
# This code was written by Noel Konagai.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, re, os, glob

from dateutil import parser
from tqdm import tqdm
from google.cloud import translate

from sklearn import preprocessing, linear_model, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# To get warnings, uncomment the below line
pd.options.mode.chained_assignment = None

# Matplotlib parameters
plt.rcParams["figure.figsize"] = [18, 10]
plt.rcParams["font.size"] = 12

def response_count(df):
    '''Counts the response rate in a given conversation'''
    all_convs = df.conv_num.unique()
    total_participants = len(df.author.unique())
    data = np.zeros((len(all_convs), 9))
    
    for i, conv in enumerate(all_convs):
        conv_df = df[df.conv_num == conv] # Sub-dataframe for the given conversation ID
        message_count = len(conv_df) # Number of messages in the given conversation
        participant_num = len(conv_df.author.value_counts()) # Number of participants in the given conversation
        response = (participant_num - 1)/total_participants # Ratio of those who participated in the conversation to the total group size
        
        # Set group type
        group_type = conv_df.group_type.iloc[0]

        # Set URL column
        url = conv_df.url.iloc[0]
        if len(url) > 2: has_url = 1
        else: has_url = 0
        
        # Set media column
        has_media = conv_df.has_media.iloc[0]
        if has_media != 1: message_length = len(str(conv_df.messages_clean.iloc[0]))
        else: message_length = 0

        # Set emoji column
        emoji = conv_df.emoji.iloc[0]
        if len(emoji) > 2: has_emoji = 1  #because empty array somehow has len of 2, probably because of \t
        else: has_emoji = 0
        
        # Set time columns
        time = parser.parse(conv_df.date_time.iloc[0])
        hour = time.hour
        day_of_week = time.weekday()

        # Write them into data
        data[i][0] = message_count
        data[i][1] = group_type
        data[i][2] = has_media 
        data[i][3] = has_url
        data[i][4] = has_emoji
        data[i][5] = message_length
        data[i][6] = hour
        data[i][7] = day_of_week
        data[i][8] = response
        
    return data

def regression_modeling(data):
    '''Models the response rate with Voting Regression'''
    # Scaling the data
    scaled_data = preprocessing.StandardScaler().fit_transform(data)

    # Creating train-test
    X = scaled_data[:,0:8]
    y = scaled_data[:,8]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Voting Regression
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()
    ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    ereg = ereg.fit(X_train, y_train)
    y_hat_ereg = ereg.predict(X_test)
    r2_ereg = r2_score(y_test, y_hat_ereg)
    return r2_ereg

def segment_by_length(df, min_length):
    '''Segments the conversation based on message character length'''
    conv_num = []
    message_lengths = []
    j = 0

    for i in range(0, len(df)):
        message = df.message.iloc[i]

        if message == "<Media omitted>":
            message_length = 0
        # Only increase conversation ID when message is longer than min_length
        else:
            message_length = len(message)
            if len(message) > min_length:
                j += 1

        if i == 0: conv_num.append(0)
        else: conv_num.append(j)
        message_lengths.append(message_length)
        
    df['conv_num'] = conv_num
    df['message_length'] = message_lengths
    
    return df

def segment_by_time(df, max_time):
    '''
    Segments conversations based on the time elapsed between two messages
    '''
    time_delta = [0.0] #List to indicate time difference between conversations
    conv_num = [0] #List to indicate conversations
    j = 0

    for i in range(1, len(df)):
        t_previous = parser.parse(df.iloc[i-1].date_time) # Timestamp of previous message
        t = parser.parse(df.iloc[i].date_time) # Timestamp of current message
        t_delta = (t - t_previous).total_seconds() / 60 # Difference of time in seconds

        if t_delta > max_time: #If it takes more than 30 mins between two messages -> assign a new conversation number
            j += 1

        time_delta.append(t_delta)
        conv_num.append(j)
        
    df['time_delta'] = time_delta
    df['conv_num'] = conv_num
    
    return df

def vary_char(char_lengths, file_path):
    '''The function loop to run to get different character lengths'''
    df = pd.read_csv(file_path, index_col = 0)
    group_names = df.group_name.unique()
    output = []

    # Loop over character length input, for every sub group
    for length in tqdm(char_lengths):
        for group in group_names:
            sub_df = df[df.group_name == group]
            sub_df = segment_by_length(sub_df, length)
            data = response_count(sub_df)
            for item in data.tolist(): 
                output.append(item) 

        # Save this dataset into a dataframe
        out_path = "../data/conversation/char/" + str(length) + ".csv"
        np_data = np.array(output)
        columns = ["message_count", "group_type", "has_media", "has_url", "has_emoji", "message_length", "hour", "day", "response_rate"]
        out_df = pd.DataFrame(data = np_data, index = None, columns = columns)
        out_df.to_csv(out_path)

        # For sanity check, display R^2 value
        r2 = regression_modeling(np_data)
        print("Char_length: {} \t r^2 val:{}".format(length, r2))

def vary_time(time_lengths, file_path):
    '''The function loop to run to get different time lengths'''
    df = pd.read_csv(file_path, index_col = 0)
    group_names = df.group_name.unique()
    output = []

    # Loop over character length input, for every sub group
    for length in tqdm(time_lengths):
        for group in group_names:
            sub_df = df[df.group_name == group]
            sub_df = segment_by_time(sub_df, length)
            data = response_count(sub_df)
            for item in data.tolist(): 
                output.append(item) 

        # Save this dataset into a dataframe
        out_path = "../data/conversation/time/" + str(length) + ".csv"
        np_data = np.array(output)
        columns = ["message_count", "group_type", "has_media", "has_url", "has_emoji", "message_length", "hour", "day", "response_rate"]
        out_df = pd.DataFrame(data = np_data, index = None, columns = columns)
        out_df.to_csv(out_path)

        # For sanity check, display R^2 value
        r2 = regression_modeling(np_data)
        print("Time_length: {} \t r^2 val:{}".format(length, r2))

def scale_data(df):
    scaled_data = preprocessing.StandardScaler().fit_transform(df)
    X = scaled_data[:,0:8]
    y = scaled_data[:,8]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def run_regression(vary_type = 'vary_time', group_breakdown = 'all'):
    if vary_type == 'vary_char':
        file_path = "../data/conversation/char/"
    elif vary_type == 'vary_time':
        file_path = "../data/conversation/time/"

    os.chdir(file_path)
    filenames = sorted([int(f[:-4]) for f in glob.glob('*.csv')])

    #Plotting variables: how the R^2 value changes
    r2_values = []
    values = []
    groups = []

    for f in filenames:
        df = pd.read_csv(str(f) + '.csv', index_col = 0)

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        if group_breakdown == 'all':
            X_train_all, X_test_all, y_train_all, y_test_all = scale_data(df)
            X_train.append(X_train_all)
            X_test.append(X_test_all)
            y_train.append(y_train_all)
            y_test.append(y_test_all)
        if group_breakdown == 'separate':
            df1 = df[df.group_type == 1]
            df2 = df[df.group_type == 2]
            df3 = df[df.group_type == 3]

            X_train_1, X_test_1, y_train_1, y_test_1 = scale_data(df1)
            X_train_2, X_test_2, y_train_2, y_test_2 = scale_data(df2)
            X_train_3, X_test_3, y_train_3, y_test_3 = scale_data(df3)

            X_train = [X_train_1, X_train_2, X_train_3]
            X_test = [X_test_1, X_test_2, X_test_3]
            y_train = [y_train_1, y_train_2, y_train_3]
            y_test = [y_test_1, y_test_2, y_test_3]

        #Voting Regression
        reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
        reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
        reg3 = LinearRegression()
        voting = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
        
        for i in range(len(X_train)):
            voting = voting.fit(X_train[i], y_train[i])
            y_hat = voting.predict(X_test[i])
            r2 = np.round(r2_score(y_test[i], y_hat), 4)

            r2_values.append(r2)
            values.append(f)

            if group_breakdown == 'all':
                output = "Function: {} \t Val: {} \t Voting Regression R^2: {} \t Groups: {}".format(vary_type, f, r2, 'all')
                groups.append('all')
            elif group_breakdown == 'separate':
                output = "Function: {} \t Val: {} \t Voting Regression R^2: {} \t Group: {}".format(vary_type, f, r2, i + 1)
                if i == 0:
                    groups.append("government")
                elif i == 1:
                    groups.append("meghshala")
                else:
                    groups.append('321')
                
            print(output)
 
    # Resetting directory
    for i in range(2):
        os.chdir("..")

    ax = sns.lineplot(x=values, y=r2_values, hue=groups, markers=True, style=groups)
    # plt.show()
    plt.title("Change in R^2 value with varying {} cutoffs".format(vary_type[5:]))
    plt.xlabel(vary_type[5:])
    plt.ylabel("R^2 value")
    plt.savefig("../figures/r2_val_{}_{}".format(vary_type,group_breakdown))

if __name__ == "__main__":
    '''vary_char 115 -> gives very negative R^2 value for Group 2, do not run it'''

    file_path = "../data/chkpt3/master_dfname.csv"
    # vary_char([91, 92, 93, 94, 95], file_path)
    # vary_time([70, 80, 90, 100, 110, 130, 140, 150, 160, 170], file_path)
    # run_regression(vary_type = 'vary_time', group_breakdown = 'separate')
    # run_regression(vary_type = 'vary_time', group_breakdown = 'all')