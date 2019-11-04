# created by Noel Konagai at 2019/11/02 11:54.
# 
# This code was written by Noel Konagai.

import argparse, os, glob, xgboost, shap
import pandas as pd
import numpy as np

from segment_conversations import segment_by_length, segment_by_time, response_count, regression_modeling
from dateutil import parser
from warnings import simplefilter

def load_files():
    os.chdir("../data/chkpt2/")
    filenames = [f for f in glob.glob("*.csv")]
    return filenames

def visualize(data, filepath):
    columns = ["message_count","num_participants", "has_media","has_url","has_emoji","message_length","hour","day","response"]
    df = pd.DataFrame(data, columns = columns)

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1] #Flip the target to be the first column
    df = df[cols]

    # train XGBoost model
    X = df.drop(columns = ["response"])
    y = df["response"]
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X)

    plt.savefig(filepath)

if __name__=="__main__":
    simplefilter(action='ignore', category=FutureWarning)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='which test to run', required=True, choices=['vary_char', 'vary_time'])
    parser.add_argument('-m', type=str, help='which regression model to run', required=False, choices=['all', 'lasso', 'decision_tree', 'ridge', 'svm', 'voting'])
    parser.add_argument('-t', type=int, help='maximum time difference to use (mins)', required=False, nargs='+')
    parser.add_argument('-l', type=int, help='minimum character length to use', required=False, nargs='+')
    parser.add_argument('-s', type=bool, help='outputs numpy file', required=False, default=False)
    parser.add_argument('-v', '--visualization', type=bool, help='outputs visualization', required=False, default=False)
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
                    png_filepath = "../figures/shap_summary_vary_char_{}.png".format(str(item))
                elif args.f == "vary_time":
                    df = segment_by_time(df, item)
                    png_filepath = "../figures/shap_summary_vary_time_{}.png".format(str(item))
                data = response_count(df)
                for data_item in data.tolist():
                    list_data.append(data_item)
            np_data = np.array(list_data)
            r2 = regression_modeling(np_data, args.m)
            print("{} regression \t {}: {} \t R^2: {} \t num_samples: {}".format(args.m, measure, item, r2, len(np_data)))
            if args.s == True:
                np_filepath = "../data/numpy/chkpt2_" + args.f + "_" + str(item) + ".npy"
                np.save(np_filepath, np_data)
            if args.v == True:
                visualize(np_data, png_filepath)

    finally:
        print('Done')
