# created by Noel Konagai, updated at 2019/11/09 10:46.
# 
# This code was written by Noel Konagai.

import pandas as pd
import numpy as np
import os, time, glob

from tqdm import tqdm
from google.cloud import translate

# Creating a client
credential_path = "../whatsapp-credentials.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# Instantiates a client
translate_client = translate.Client()

def translate_message(df, filename):
    '''
    Creates a translated message language column in the dataframe
    '''
    language_col = [] #language code
    translated_col = [] #translated message

    print("Processing ", filename)

    for _, message in tqdm(enumerate(df.messages_clean)):
        # Catching any nan values
        if str(message) == "nan":
            language_col.append("")
            translated_col.append("")
        else:
            # Catching media omissions
            if message == "<Media omitted>":
                language_col.append("")
                translated_col.append("")
                
            # Otherwise, determines the language of the message
            # gets its confidence, and translates it to EN
            else:
                result = translate_client.translate(message, target_language="en")
                language_col.append(result['detectedSourceLanguage'])
                translated_col.append(result['translatedText'])
        time.sleep(0.01)

    df['translated_message'] = translated_col
    df['language'] = language_col

    return df

def get_filenames(path, extension):
    os.chdir(path)
    filenames = [f for f in glob.glob(extension)]
    os.chdir("../")
    return filenames

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df
    
if __name__ == "__main__": 
    input_path = "../data/chkpt1/"
    output_path = "../data/chkpt2/"
    filenames = get_filenames(input_path, "*.csv")
    chkpt2_filenames = get_filenames(output_path, "*.csv")


    for filename in filenames:
        out_filename = filename[:-5] + "2.csv"
        
        #Only process and save file if not already translated
        if out_filename not in chkpt2_filenames:
            df = read_csv(input_path + filename)
            df_translated = translate_message(df, filename)
            df_translated.to_csv(output_path + out_filename)