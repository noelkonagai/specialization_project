# created by Noel Konagai at 2019/10/28 14:56.
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

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def segment_by_time(df, max_time):
    '''
    A conversation is defined as two messages do not have more than max_time minutes elapsed
    between each other. Creates a new column in DF with conversation ID. 
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

def segment_by_length(df, min_length):
    '''
    A conversation is defined as a streak of messages that started with a message of at least length min_length.
    '''
    conv_num = [] #List to indicate conversations
    message_lengths = []
    j = 0

    for i in range(len(df)):
        message = df.message.iloc[i]

        if message == "<Media omitted>":
            message_length = 0
        # Only increase conversation ID when message is longer than min_length
        else:
            message_length = len(message)
            if len(message) > min_length:
                j += 1

        conv_num.append(j)
        message_lengths.append(message_length)
        
    df['conv_num'] = conv_num
    df['message_length'] = message_lengths
    
    return df

# def retrieve_conversation(df, conv_num):
#     '''
#     Function used to get messages with a given conversation ID
#     '''
#     conversation = list(df[df['conv_num'] == conv_num]['translated_message'])
    
#     for i in range(len(conversation)):
#         print(str(conversation[i]))
    
#     return conversation

def translate_message(message):
    # Creating a client
    credential_path = "../whatsapp-credentials.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    # Instantiates a client
    translate_client = translate.Client()

    result = translate_client.translate(message, target_language="en")
    src_lng = result['detectedSourceLanguage']
    message_translated = result['translatedText']
    
    return message_translated, src_lng

# def get_sentiment(message):
#     analyzer = SentimentIntensityAnalyzer()
#     for sentence in sentences:
#         vs = analyzer.polarity_scores(sentence)
#         print("{:-<65} {}".format(sentence, str(vs)))

def response_count(df):
    '''
    Creates a numpy matrix with the following structure
    X (columns 0 - 3):  num of messages for a conversationID,
                        participants in conversation,
                        hour of the day message is sent,
                        day of the week message is sent,
    y (column 4):       rate of non-response
    '''
    total_convs = len(df.conv_num.value_counts())
    total_participants = len(df.author.value_counts())
    data = np.zeros((total_convs, 9))
    
    for i in range(total_convs):
        conv_df = df[df.conv_num == i] # Sub-dataframe for the given conversation ID
        message_count = len(conv_df) # Number of messages in the given conversation
        participant_num = len(conv_df.author.value_counts()) # Number of participants in the given conversation
        response = (participant_num - 1)/total_participants # Ratio of those who participated in the conversation to the total group size
        # group_type = conv_df.group_type.iloc[0]

        url = conv_df.url.iloc[0]
        if len(url) > 2:
            has_url = 1
        else:
            has_url = 0
        
        has_media = conv_df.has_media.iloc[0]

        if has_media != 1:
            message_length = len(str(conv_df.messages_clean.iloc[0]))
        else:
            message_length = 0

        emoji = conv_df.emoji.iloc[0]

        if len(emoji) > 2: #because empty array somehow has len of 2, probably because of \t
            has_emoji = 1
        else:
            has_emoji = 0
        
        time = parser.parse(conv_df.date_time.iloc[0])
        hour = time.hour
        day_of_week = time.weekday()
        
        data[i][0] = message_count
        data[i][1] = 0
        data[i][2] = has_media 
        data[i][3] = has_url
        data[i][4] = has_emoji
        data[i][5] = message_length
        data[i][6] = hour
        data[i][7] = day_of_week
        data[i][8] = response
        
    return data

def get_filenames(path):
    os.chdir(path)
    filenames = [f for f in glob.glob("*.csv")]
    os.chdir("..")
    return filenames

def concatenate_data(list_data):
    concat_data = np.concatenate(list_data[0], list_data[1], axis=0)

    for i in range(2, len(list_data)):
        concat_data = np.concatenate(concat_data, list_data[i], axis=0)
    
    return concat_data

def regression_modeling(data, model):
    # Scaling the data
    scaled_data = preprocessing.StandardScaler().fit_transform(data)

    # Creating train-test
    X = scaled_data[:,0:8]
    y = scaled_data[:,8]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = [  ['linear', linear_model.Lasso(alpha=0.1).fit(X_train, y_train)],
                ['decision_tree' , DecisionTreeRegressor(random_state=0).fit(X_train, y_train)],
                ['ridge', linear_model.Ridge(alpha=.5).fit(X_train, y_train)],
                ['svm', svm.SVR(kernel='rbf', gamma='auto').fit(X_train, y_train)]]

    if model == 'all':
        for m in models:
            y_predicted = m[1].predict(X_test)
            r2 = r2_score(y_test, y_predicted)
            print("{}: {}".format(m[0], r2))

    if model == "lasso":
        lasso_reg = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
        y_hat_lasso = lasso_reg.predict(X_test)
        r2_lasso = r2_score(y_test, y_hat_lasso)
        print("R^2 score for Lasso:", r2_lasso)

    elif model == "decision_tree":
        dt_reg = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
        y_hat_dt = dt_reg.predict(X_test)
        r2_dt_reg = r2_score(y_test, y_hat_dt)
        print("R^2 score for Decision tree:", r2_dt_reg)

    elif model == "ridge":
        ridge_reg = linear_model.Ridge(alpha=.5).fit(X_train, y_train)
        y_hat_ridge = ridge_reg.predict(X_test)
        r2_ridge = r2_score(y_test, y_hat_ridge)
        print("R^2 score for Ridge:", r2_ridge)

    elif model == "svm":
        svm_reg = svm.SVR(kernel='rbf').fit(X_train, y_train)
        y_hat_svm = svm_reg.predict(X_test)
        r2_svm = r2_score(y_test, y_hat_svm)
        print("R^2 score for RBF SVM:", r2_svm)

    elif model == "voting":
        reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
        reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
        reg3 = LinearRegression()
        ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
        ereg = ereg.fit(X_train, y_train)
        y_hat_ereg = ereg.predict(X_test)
        r2_ereg = r2_score(y_test, y_hat_ereg)
        return r2_ereg

def plot_message_length(input_path, filenames):
    '''
    With an input of a list containing filenames, will return a series and a plot
    that visualizes the distribution of message lengths.
    '''
    lengths = []
    n_bins = 50

    for filename in tqdm(filenames):
        df = pd.read_csv(input_path + filename)
        message_column = df.translated_message
        for message in list(message_column):
            length = len(str(message))
            if length <= 100:
                lengths.append(length)

    fig = plt.hist(lengths, bins=n_bins)
    plt.savefig("../figures/message_length_histogram.png")
    plt.show()

    np.save("../numpy_data/message_length", np.array(lengths))

    return lengths, fig

def varying_char_length(char_lengths):

    list_data = []

    # Loop over character length input
    for length in char_lengths:
        for filename in filenames:
            file_path = input_path +  filename
            df = pd.read_csv(file_path)
            df = segment_by_length(df, length)
            data = response_count(df)
            for item in data.tolist():
                list_data.append(item) 

        # Save this dataset into a dataframe
        filepath = "../data/conversation/length/" + str(length) + ".csv"
        numpy_list_data = np.array(list_data)
        columns = ["message_count", "group_type", "has_media", "has_url", "has_emoji", "message_length", "hour", "day", "response_rate"]
        df_output = pd.DataFrame(data = numpy_list_data, index = None, columns = columns)
        df_output.to_csv(filepath)

        r2_val = regression_modeling(numpy_list_data, 'voting')
        print("Char_length: {} \t r^2 val:{}".format(length, r2_val))

if __name__ == "__main__":
    input_path = "../data/chkpt3/"
    filenames = get_filenames(input_path)

    varying_char_length([80, 82, 84, 86, 88, 90])

    ''' TO DO!
    - Log regression R^2 values into a file
    - Include sentiment column of message + another column on emoji
    - Experiment with weekdays as their own columns (0/1)
    '''