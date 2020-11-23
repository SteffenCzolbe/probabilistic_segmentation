import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
import pickle


def load_data(test_results_file, dataset):
    MAX_POINTS_PER_CLASS = 3000
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    models = test_results[dataset].keys()
    data = defaultdict(list)
    median = defaultdict(list)
    for model in models:
        for condition in ['tp', 'fp', 'fn', 'tn']:
            uncertainties = test_results[dataset][model][
                'per_sample'][f"test/{condition}_uncertainty"]

            if len(uncertainties) > MAX_POINTS_PER_CLASS:
                uncertainties = np.random.choice(
                    uncertainties, size=MAX_POINTS_PER_CLASS, replace=False)
            model_name = test_results[dataset][model]['model_name']
            data['uncertainty'] += list(uncertainties)
            median['uncertainty'].append(np.median(uncertainties))
            data['model'] += [model_name for _ in range(len(uncertainties))]
            median['model'].append(model_name)
            data['condition'] += [condition.upper()
                                  for _ in range(len(uncertainties))]
            median['condition'].append(condition.upper())

    # dataframe columns: #ged,  #samples, model
    return pd.DataFrame(data), pd.DataFrame(median)


def main(args):
    # load data  minto pandas dataframes
    df_data, df_median = load_data(args.test_results_file, args.dataset)
    sns.set_theme(style="whitegrid")

    # plot datapoints
    ax = sns.stripplot(x='condition', y="uncertainty", hue="model",
                       data=df_data, palette="Set2", dodge=True,
                       size=1, jitter=0.2)

    # plot means on top
    ax = sns.stripplot(x='condition', y="uncertainty", hue="model",
                       data=df_median, palette="Set2", dodge=True,
                       size=9, jitter=0, edgecolor='black', ax=ax, linewidth=1)

    # Get the handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first 4 elements
    l = plt.legend(handles[:4], labels[:4])

    # y-label
    ax.set_ylabel('Uncertainty $H$')

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