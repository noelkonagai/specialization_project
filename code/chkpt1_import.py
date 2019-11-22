# created by Noel Konagai at 2019/10/28 14:56.
# 
# This code was written by Noel Konagai.

import pandas as pd
import numpy as np
import re, emoji, regex, datetime, os, glob, dateparser

from dateutil import parser
from sklearn.preprocessing import LabelEncoder
from urlextract import URLExtract
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

def create_csv(data_dir, filename):
    '''
    Creates a CSV file with the input of a data_dir where the .txt WhatsApp
    conversation file is located. Requires a filename input without the .txt extension. 
    Saves the WhatsApp conversation history in a .csv file and returns the dataframe.
    '''
    inpath = data_dir + filename + ".txt"
    
    f = open(inpath, 'r')
    text = f.read()
 
    # Every text message has the same format: date - sender: message. 
    messages = re.findall('(.*?) - (.*?): (.*)', text)

    #Convert list to a dataframe and name the columns
    history = pd.DataFrame(messages, columns = ['date_time','author','message'])
    outpath = data_dir + filename + ".csv"
    history.to_csv(outpath, index = False)

def clean_original(df_original):
    #Execute only once to get the cleaned Dataframe
    #This is only used for Rama's cleaned files
    df = df_original.copy()
    df.columns = ['date_time','sender','sender_role',
                'message', 'category']
    return df

def convert_date(date_time):
    '''
    Converts date_time into machine readable timestamps
    '''
    return parser.parse(date_time.strip('\t'))

def demojize(text):
    '''
    Removes Emojis from a given text
    '''
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

def split_count(text):
    '''
    Creates a list of emojis from a given text
    '''
    emoji_list = []
    data = regex.findall(r'\X', text)
    
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list

def find_url(text):
    '''
    Creates a list of URLs found in a given text
    '''
    extractor = URLExtract()
    return extractor.find_urls(text)

def main(filename):
    '''
    The main cleaning pipeline given a filename without .csv extension
    '''
    filepath = data_dir + filename + '.csv'
    
    # Importing csv to dataframe
    df = pd.read_csv(filepath)
    
    # Creating a proper time stamp
    df.date_time = df.date_time.apply(convert_date)
    
    # Encoding senders into sender IDs
    le = LabelEncoder()
    le.fit(df['author'])
    df['sender_id'] = le.transform(df['author'])

    # Creates a dictionary of sender ids mapped to sender nums
    sender_ids = le.transform(le.classes_)
    sender_nums = le.inverse_transform(sender_ids)
    le_name_mapping = dict(zip(sender_ids, sender_nums))

    # Extracting media, emojis, and URLs
    message_series = [] #Original messages with empty strings at group activities
    em_series = [] #List of emojis
    demojized_series = [] #Cleaned message devoid of emojis
    url_series = [] #List of URLs found in the message
    media_binary_series = [] #Binary encoding whether the message contains a media file

    for i in range(len(df.message)):
        if i % 100 == 0:
            print("Now doing {}th iteration.".format(i))

        if pd.isnull(df.message.iloc[i]) == False:
            msg = df.message.iloc[i]
            em_list = split_count(df.message.iloc[i])
            dmj = demojize(df.message.iloc[i])
            url = find_url(df.message.iloc[i])
            if df.message.iloc[i] == "<Media omitted>":
                media = 1
            else:
                media = 0

        else:
            msg = ""
            em_list = []
            dmj = ""
            url = []
            media = 0

        message_series.append(msg)
        em_series.append(em_list)
        demojized_series.append(dmj)
        url_series.append(url)
        media_binary_series.append(media)

    df['emoji'] = em_series
    df['messages_clean'] = demojized_series
    df['url'] = url_series
    df['has_media'] = media_binary_series
    
    return df

def get_filenames(path, extension):
    os.chdir(path)
    filenames = [f for f in glob.glob(extension)]
    os.chdir("..")
    return filenames

if __name__ == "__main__":
    # filenames = ["belgavi",
    #         "chikabagawadi",
    #         "hindalga",
    #         "marihal",
    #         "nallikalli",
    #         "smart_hms",
    #         "subramanya",
    #         "suliya"]

    data_dir = "../data/raw/"
    out_dir = "../data/chkpt1/"

    filenames = get_filenames(data_dir, "*.txt")

    print("Creating CSVs of the raw .txt files")
    for filename in tqdm(filenames):
        # Only create CSVs if it is not in the data directory
        csv_filename = filename[:-4] + ".csv"
        if csv_filename not in get_filenames(data_dir, "*.csv"):
            create_csv(data_dir, filename[:-4])

    #not_working = ['rosemary.txt', 'hindalga.txt', 'marihal.txt', 'chikabagawadi.txt', 'meghshala-mysore.txt', 'meghshala-belgaum-1.txt']
    not_working = []
    chkpt1_filenames = get_filenames(out_dir, "*.csv")

    print("Cleaning CSV files")
    for filename in tqdm(filenames): #Iterating over .txt extension filenames
        csv_chkpt1_filename = filename[:-4] + "_chkpt1.csv" #removing .txt, append _chkpt1.csv
        #Only create files if it has not been made before, and it is not amongst files that do not work
        if filename in not_working:
            print("This file has errors: ", filename[:-4])
        elif csv_chkpt1_filename in chkpt1_filenames:
            print("This file is already in output directory: ", filename[:-4])
        else:
            print("Now cleaning: ", filename[:-4])
            df = main(filename[:-4])
            # out_path = out_dir + filename[:-4] + "_chkpt1_" + datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".csv"
            out_path = out_dir + filename[:-4] + "_chkpt1.csv"
            df.to_csv(out_path, index = False)

    '''
    Known issues:
    - rosemary.txt creates an empty .csv file
    - hindalga.txt creates a file that the next .py file can't read
    '''