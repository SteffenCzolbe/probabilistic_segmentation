import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
import pickle


def load_data(test_results_file, dataset):
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    models = test_results[dataset].keys()
    data = defaultdict(list)
    median = defaultdict(list)
    for model in models:
        corr = test_results[dataset][model]['per_sample']["test/uncertainty_seg_error_correl"]
        nan_vals = np.count_nonzero(np.isnan(corr))
        if nan_vals > 0:
            print(
                f'WARNING: Ignoring {nan_vals}/{len(corr)} Nan-Values during plotting! Probably caused by gt annotation being 100% background.')
        model_name = test_results[dataset][model]['model_name']
        data['corr'] += list(corr)
        median['corr'].append(np.nanmedian(corr))
        data['model'] += [model_name for _ in range(len(corr))]
        median['model'].append(model_name)
        # we add a dummy x-axis
        data['x'] += [0 for _ in range(len(corr))]
        median['x'].append(0)

    # dataframe columns: corr,  model, x
    return pd.DataFrame(data), pd.DataFrame(median)


def main(args):
    # load data  minto pandas dataframes
    df_data, df_median = load_data(args.test_results_file, args.dataset)
    sns.set_theme(style="whitegrid")

    # plot datapoints
    ax = sns.stripplot(x='x', y="corr", hue="model",
                       data=df_data, palette="Set2", dodge=True,
                       size=1.3, jitter=0.2)

    # plot means on top
    ax = sns.stripplot(x='x', y="corr", hue="model",
                       data=df_median, palette="Set2", dodge=True,
                       size=9, jitter=0, edgecolor='black', ax=ax, linewidth=1)

    # Get the handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, dont use duplicate entries
    n = len(handles) // 2
    l = plt.legend(handles[:n], labels[:n])

    # hide dummy-x acis label
    plt.xticks([])
    ax.set_xlabel(None)

    # y-label
    ax.set_ylabel('Corr$(H, $seg_error$)$')

    # save
    fig = ax.get_figure()
    for f in args.output_file:
        fig.savefig(f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, help='Dataset to plot')
    parser.add_argument(
        '--test_results_file', type=str, default='plots/experiment_results.pickl', help='File with test results.')
    parser.add_argument(
        '--output_file', type=str, nargs="+", help='File to save the results in.')
    args = parser.parse_args()
    main(args)
