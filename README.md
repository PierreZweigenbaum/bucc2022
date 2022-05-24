# bucc2022
Code for the __BUCC 2022 Shared Task on bilingual term alignment in comparable corpora__.

## Computing evaluation scores, including uninterpolated Average Precision

evaluation.py compares each system run (*.txt) to the gold standard (-g) and compute average precision according to the formula.  With option -i, when reading system runs, ignores source terms not in the source list (-s) and target terms not in target list (-t):

```
    $ python evaluation.py -g terms-en-fr.txt -s terms-en.txt -t terms-fr.txt -i SYSTEM1/*.txt >eval-i.tsv
```

Additionally, --interpolated computes interpolated average precision, and --details tabulates the evolution of various scores along with true positives:

```
    $ python evaluation.py -g terms-en-fr.txt -s terms-en.txt -t terms-fr.txt -i --interpolated --details SYSTEM1/*.txt >eval-i-inter-details.tsv
```

## Plotting Average Precision as a function of ranked results

plot_scores.py plots scores as a function of true positives in the order returned by a system.  It takes as input the detailed table produced above with --interpolated --details.  The data points with false positives are ignored, only data points with true positives are plotted.

```
	$ python plot_scores.py  eval-i-inter-details.tsv --top-col k --scores AP_u_min P R F1
```
