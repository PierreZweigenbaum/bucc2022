#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Evaluate BUCC 2022 submission against gold dictionary.
System results are a ranked list of term pairs.
Computes Average Precision: each predicted term pair is considered as a retrieved document,
and is considered relevant if it is present in the gold dictionary.
Outputs a table in tab-separated format.

Example:

  $ python ../bin/evaluation.py -g ../bucc2022_test_enfr/terms-en-fr.txt -s ../bucc2022_test_enfr/terms-en.txt -t ../bucc2022_test_enfr/terms-fr.txt -i SYSTEM1/*.txt >eval-i.tsv

  compares each system run (*.txt) to the gold standard (-g) and computes average precision according to the formula.
  -i: In a system run, ignores source terms not in source list (-s) and target terms not in target list (-t).

  $ python ../bin/evaluation.py -g ../bucc2022_test_enfr/terms-en-fr.txt -s ../bucc2022_test_enfr/terms-en.txt -t ../bucc2022_test_enfr/terms-fr.txt -i --interpolated --details SYSTEM1/*.txt >eval-i.tsv

  additionally (--interpolated) computes interpolated average precision, and (--details) tabulates the evolution of various scores along with true positives
"""

"""
    Copyright (c) 2022 LISN CNRS
    All rights reserved.
    Pierre Zweigenbaum, LISN, CNRS, Université Paris-Saclay <pz@lisn.fr>
"""

# Program version
version = '1.1'

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd

def _format_int_or_float(val, float_format="{:.3f}", int_format="{}"):
    if isinstance(val, float):
        return float_format
    elif type(val) is int:
        return int_format
    else:
        return "{}"

def _allowed_possible_pairs(source, target, targets_per_source):
    "Total number of possible source, target terms.  Currently unconstrained."
    return int(targets_per_source * (len(source) + len(target)) / 2)

def p_r_f1(tp, fp, fn):
    s = tp + fp
    g = tp + fn
    p = (0 if s == 0 else tp / s) # 1 if s == 0 and g == 0 ?
    r = (1 if g == 0 else tp / g)
    f1 = (0 if tp == 0 else 2 * p * r / (p + r))
    return p, r, f1

def set_measures(system, gold):
    s = len(system)
    g = len(gold)
    hits = [ 1*(p in gold) for p in system ] # system hits among ranked documents
    tp = np.sum(hits)
    fp = s - tp
    fn = g - tp
    p, r, f1 = p_r_f1(tp, fp, fn)
    return [s, g, tp, fp, fn, p, r, f1]

def average_precision_uninterpolated(system, gold):
    """AP = 1/m ∑_(k = 1)^m P(R_k)"""
    m = len(gold)
    sigma_p_k = 0
    tp = 0
    for k, d in enumerate(system):
        if d in gold:
            tp += 1
            p_k = float(tp) / (k+1)
            sigma_p_k += p_k

    return sigma_p_k / m

def average_precision_interpolated(system, gold):
    """AP = 1/m ∑_(k = 1)^m P(R_k)"""
    m = len(gold)
    tot_tp = len(gold.intersection(system))
    tp, fp = 0, 0
    sigma_p_k = 0.0
    # k_tp_fp_p_ap_ap = np.zeros((m, 6))
    cols = { "k": np.zeros(tot_tp, dtype=int), "tp": np.zeros(tot_tp, dtype=int),
             "fp": np.zeros(tot_tp, dtype=int),
             "p": np.zeros(tot_tp), "p_i": np.zeros(tot_tp),
             "apu_min": np.zeros(tot_tp), "apu_max": np.zeros(tot_tp),
             "api_min": np.zeros(tot_tp), "api_max": np.zeros(tot_tp),
             "pair": tot_tp * [""]}
    df = pd.DataFrame(cols)
    # uninterpolated p_k
    for i, d in enumerate(system):
        if d in gold:
            tp += 1
            p_k = float(tp) / (i+1)
            sigma_p_k += p_k
            df.loc[tp-1, ["k", "tp", "fp", "p", "p_i", "apu_min", "apu_max", "pair"]] = [i+1, tp, fp, p_k, p_k, sigma_p_k/m, sigma_p_k/tp, d]
        else:
            fp += 1

    ap_u = sigma_p_k / m

    # _interpolate_p_list(k_tp_fp_p_ap_ap[:,3])
    _interpolate_p_list(np.asarray(df["p_i"])) # modify as side-effect

    sigma_p_k = 0.0
    for i in range(tp):
        # (k, tp_k, ignore, p_k, ignore, ignore) = k_tp_fp_p_ap_ap[i]
        sigma_p_k += df["p_i"][i]
        tp_k = df["tp"][i]
        df.loc[i, ["api_min", "api_max"]] = [sigma_p_k/m, # current min ap
                                         sigma_p_k/tp_k] # current max ap

    ap_i = sigma_p_k / m

    return ap_u, ap_i, df

def _interpolate_p_list(p_l):
    """Interpolated precision at position i: max(p) for j >= i.

    Implement by walking list from right to left,
    keeping track of max value so far,
    and replacing values lower than max with max.
    Modifies p_l IN PLACE."""
    max = 0
    new_l = []
    for i in range(1, len(p_l)+1):
        p = p_l[-i]
        if p > max:
            max = p
        elif p < max:
            p_l[-i] = max
    return p_l

def f_read_list(f):
    logging.info(f"  Reading file '{f} as a list of lines'")
    with open(f, "r", encoding="utf8") as fs:
        return [ s.strip() for s in fs.readlines() ]

def filter_oov(l, source, target, file=None):
    if file is not None:
        logging.info(f"  Filtering OOVs in '{file}' with {len(source)} source terms and {len(target)} target terms")
    filtered = []
    for pair in l:
        s, t = pair.split("\t")
        if s in source and t in target:
            filtered.append(pair) # keep the string
    logging.info(f"    -> from {len(l)} to {len(filtered)} term pairs")
    return filtered

#================
if __name__ == '__main__':
    def parse_execute_command_line():
        parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=__doc__)

        groupI = parser.add_argument_group('Input files')
        groupI.add_argument('system', nargs='+',
                            help="file(s) containing system prediction: an ordered list of tab-separated term pairs")
        groupI.add_argument('-g', '--gold', required=True,
                            help="file containing gold standard: a list of tab-separated term pairs")
        groupI.add_argument('-s', '--source-terms', help="file containing source terms, one per line")
        groupI.add_argument('-t', '--target-terms', help="file containing target terms, one per line")

        groupP = parser.add_argument_group('Parameters')
        groupP.add_argument('--interpolated', action='store_true', help="also compute interpolated precision")
        groupP.add_argument('-d', '--details', action='store_true', help="report details of computation of average precision (only if interpolated precision is computed)")
        groupP.add_argument('-i', '--ignore-oov', action='store_true', help="ignore out-of-source-or-target terms; requires to provide --source and --target term lists")

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

        if args.ignore_oov and not (args.source_terms and args.target_terms):
            logging.error(f"The option --ignore-oov requires the specification of --source-terms and --target-terms")

        gold = set(f_read_list(args.gold))
        n_gold = len(gold)
        gold_s = { g.split("\t")[0] for g in gold } # source terms in gold
        gold_t = { g.split("\t")[1] for g in gold } # target terms in gold (+ "\n")
        logging.info(f"Gold: {n_gold} distinct pairs, {len(gold_s)} source terms, {len(gold_t)} target terms")

        source = set(f_read_list(args.source_terms))
        target = set(f_read_list(args.target_terms))
        logging.info(f"Term lists: {len(source)} distinct source terms, {len(target)} distinct target terms")

        systems = [ (s, f_read_list(s)) for s in args.system ]
        if args.ignore_oov:
            systems = [ (s, filter_oov(l, source, target, file=s)) for s, l in systems ]

        columns = ["run",
                   "k",
                   "AP_u_min", "AP_i_min", "AP_u_max", "AP_i_max",
                   "nSys", "nGold", "TP", "FP", "FN",
                   "P", "R", "F1", "P_i", "source", "target"]
        print("\t".join(columns))

        for s, ordered_pairs in systems:

            scores2 = set_measures(ordered_pairs, gold)
            logging.info(f"  sys={s}, nb_pairs={len(ordered_pairs)}, scores2={[ _format_int_or_float(e).format(e) for e in scores2 ]}")
            if args.interpolated:
                map_u, map_i, df = average_precision_interpolated(ordered_pairs, gold)
                if args.details:
                    m = len(gold)
                    for i in range(len(df)):
                        (k, tp_k, fp_k, intp_k, min_apu_k, max_apu_k, min_api_k, max_api_k, pair) = df.loc[i, ["k", "tp", "fp", "p_i", "apu_min", "apu_max", "api_min", "api_max", "pair"]]
                        fn_k = m - tp_k
                        p_k, r_k, f1_k = p_r_f1(tp_k, fp_k, fn_k)
                        print("\t".join([ s, str(int(k)) ]
                                        + [ f"{e:.4f}" for e in (min_apu_k, min_api_k, max_apu_k, max_api_k) ]
                                        + [ str(int(e)) for e in (k, m, tp_k, fp_k, fn_k) ]
                                        + [ f"{e:.4f}" for e in (p_k, r_k, f1_k, intp_k) ]
                                        + [pair]))
            else:
                map_i = 0.0
                map_u = average_precision_uninterpolated(ordered_pairs, gold)
            print("\t".join([s, str(len(ordered_pairs))]
                            + [ _format_int_or_float(e, float_format="{:.4f}").format(e) for e in
                                [map_u, map_i, map_u, map_i, *scores2, scores2[5]] ]
                            + ["_mean_\t_mean_"]))

    parse_execute_command_line()
