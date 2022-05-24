#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Evaluation of BUCC 2022 submission: plotting scores.
Requires pandas version closer to 1.1.5 than to 1.4.1.

Given a table with scores for top-k ranked retrieved documents,
plots scores for each run in the table.

Example:
  $ python plot_scores.py eval-scores-i-inter-details.tsv --top-col k --scores AP_u_min P R F1

  creates file eval-runs-i-inter-details.pdf

See evaluation.py to create the .tsv scores file.

"""

"""
    Copyright (c) 2022 LISN CNRS
    All rights reserved.
    Pierre Zweigenbaum, LISN, CNRS, Université Paris-Saclay <pz@lisn.fr>
"""

# Program version
version = '0.1'

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def output_file_name(file, old=".tsv", new=".pdf"):
    old_l = len(old)
    if (len(file) > old_l) and (file[-old_l:] == old):
        file = file[:-old_l]
    return file + new

def plot_scores(file, output_type=".pdf", cols=["AP", "P", "R"], run_col="run", top_col="top", base_height=4):
    df = pd.read_csv(file, delimiter="\t")

    runs = np.unique(df[run_col])
    columns = df.columns.values
    logging.info(f"runs = {runs}, run_col='{run_col}', top_col='{top_col}', cols='{cols}'")
    assert all( c in columns for c in cols ), f"The specified --scores {cols} must be column names in the input file, i.e., {columns}"

    fig, ax = plt.subplots(1, len(cols), figsize=(base_height * len(cols), base_height))

    for i, r in enumerate(runs):
        run = df[df[run_col]==r]
        logging.info(f"  Plots of x='{top_col}' vs y={cols} for system {r}: {len(run[[top_col]])} rows")

        for j, col in enumerate(cols):
            # this works with pandas 1.1.5, not with pandas 1.4.1:
            ax[j].plot(run[[top_col]], run[[col]], lw=3)
            ax[j].set_ylabel(col)
            ax[j].set_xlabel('ranked term pairs')

    fig.legend(runs, loc='upper center')
    fig.tight_layout()
    plt.savefig(output_file_name(file, ".tsv", output_type))

#================
if __name__ == '__main__':
    def parse_execute_command_line():
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=__doc__)

        groupI = parser.add_argument_group('Input files')
        groupI.add_argument('input_file', help="file systems scores: tab-separated rows. Must include 'run' and 'top' columns.")

        groupP = parser.add_argument_group('Parameters')
        groupP.add_argument('--scores', nargs="+", default=['AP', 'P', 'R'], help="list of scores. Default: '%(default)s'. These scores must be column names in the input file.  One sub-figure is created for each score.")
        groupP.add_argument('--run-col', default="run", help="name of column for run name. Default: '%(default)s'. In each plot, there is one graph for each unique value in this column.")
        groupP.add_argument('--top-col', default="top", help="name of column for top value. Default: '%(default)s'. This provides the x value in each plot.")
        groupP.add_argument('--output-extension', default=".pdf", help="extension of output figure, determines the file type. Default: '%(default)s'.")

        groupS = parser.add_argument_group('Special')
        groupS.add_argument("-q", "--quiet", action="store_true", help="suppress reporting progress info.")
        groupS.add_argument("--debug", action="store_true", help="print debug info.")
        groupS.add_argument("-v", "--version", action="version", version='%(prog)s ' + version, help="print program version.")

        args = parser.parse_args()

        FORMAT = '%(levelname)s: %(message)s'
        logging.basicConfig(format=FORMAT)

        logger = logging.getLogger()
        if not args.quiet:
            logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)

        logging.info(f"Plotting x='{args.top_col}' against y for y in {args.scores} for each unique system in column '{args.run_col}' of table '{args.input_file}'")
        plot_scores(args.input_file, output_type=args.output_extension, cols=args.scores, run_col=args.run_col, top_col=args.top_col)

    parse_execute_command_line()
