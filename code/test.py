# created by Noel Konagai at 2019/11/02 11:54.
# 
# This code was written by Noel Konagai.

import argparse, os, glob
import pandas as pd
import numpy as np
from segment_conversations import segment_by_length, segment_by_time, response_count, regression_modeling
from dateutil import parser
from warnings import simplefilter

def load_files():
    os.chdir("../data/chkpt2/")
    filenames = [f for f in glob.glob("*.csv")]
    return filenames

if __name__=="__main__":
    simplefilter(action='ignore', category=FutureWarning)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='which test to run', required=True, choices=['vary_char', 'vary_time'])
    parser.add_argument('-m', type=str, help='which regression model to run', required=False, choices=['all', 'lasso', 'decision_tree', 'ridge', 'svm', 'voting'])
    parser.add_argument('-t', type=int, help='what time difference to use (mins)', required=False, nargs='+')
    parser.add_argument('-l', type=int, help='what minimum character length to use', required=False, nargs='+')
    parser.add_argument('-s', type=bool, help='Do you want to save the output to numpy? Required for further visualization.', required=False, default=False)
    args = parser.parse_args()

    print("\nRunning test: {}".format(args.f))

    try:

        if args.f == "vary_char":
            iterator = args.l
            measure = "char_length"
        elif args.f == "vary_time":
            iterator = args.t
            measure = "time_length"

        for item in iterator:
            filenames = load_files()
            list_data = []
            for filename in filenames:
                file_path = "../data/chkpt2/" + filename
                df = pd.read_csv(file_path)
                if args.f == "vary_char":
                    df = segment_by_length(df, item)
                elif args.f == "vary_time":
                    df = segment_by_time(df, item)
                data = response_count(df)
                for data_item in data.tolist():
                    list_data.append(data_item)
            np_data = np.array(list_data)
            r2 = regression_modeling(np_data, args.m)
            print("{} regression \t {}: {} \t R^2: {} \t num_samples: {}".format(args.m, measure, item, r2, len(np_data)))
            if args.s == True:
                np_filepath = "../data/numpy/chkpt2_" + args.f + "_" + str(item) + ".npy"
                np.save(np_filepath, np_data)
    finally:
        print('Done')
