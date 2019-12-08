# created by Noel Konagai at 2019/12/08 13:13.
# 
# This code was written by Noel Konagai.

import argparse
from chkpt3_analysis import vary_char, vary_time, run_regression
from warnings import simplefilter

file_path = "../data/chkpt3/master_dfname.csv"

if __name__=="__main__":
    simplefilter(action='ignore', category=FutureWarning)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='which test to run', required=False, choices=['vary_char', 'vary_time'])
    parser.add_argument('-t', type=int, help='maximum time difference to use (mins)', required=False, nargs='+')
    parser.add_argument('-c', type=int, help='minimum character length to use', required=False, nargs='+')
    parser.add_argument('-r', type=str, help='choose whether to run regressions groups separately or together', required=False, choices =['separate', 'all'])
    parser.add_argument('-d', type=str, help='choose the type of data to use for regression', required=False, choices=['vary_char', 'vary_time'])
    args = parser.parse_args()

    print("\nRunning test: {}".format(args.f))

    try:
        if args.f == "vary_char":
            vary_char(args.c, file_path)
        elif args.f == "vary_time":
            vary_time(args.t, file_path)
    
        if args.r:
            run_regression(args.d, args.r)

    finally:
        print('Done')
