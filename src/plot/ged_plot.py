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
        for sample_count in 1, 4, 8, 16:
            geds = test_results[dataset][model]['per_sample'][f"test/ged/{sample_count}"]
            model_name = test_results[dataset][model]['model_name']
            data['ged'] += list(geds)
            median['ged'].append(np.median(geds))
            data['model'] += [model_name for _ in range(len(geds))]
            median['model'].append(model_name)
            data['num. samples'] += [sample_count for _ in range(len(geds))]
            median['num. samples'].append(sample_count)

    # dataframe columns: #ged,  #samples, model
    return pd.DataFrame(data), pd.DataFrame(median)


def main(args):
    # load data  minto pandas dataframes
    df_data, df_median = load_data(args.test_results_file, args.dataset)
    sns.set_theme(style="whitegrid")

    # set-up figure
    fig, ax = plt.subplots(figsize=(5*0.9, 4*0.9))

    # plot datapoints
    sns.stripplot(x='num. samples', y="ged", hue="model",
                  data=df_data, palette="Set2", dodge=True,
                  size=1, jitter=0.2, ax=ax)

    # plot means on top
    sns.stripplot(x='num. samples', y="ged", hue="model",
                  data=df_median, palette="Set2", dodge=True,
                  size=8, jitter=0, edgecolor='black', ax=ax, linewidth=1)

    if args.legend:
        # Get the handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, dont use duplicate entries
        n = len(handles) // 2
        l = plt.legend(handles[:n], labels[:n])
    else:
        ax.get_legend().remove()

    # y-label
    ax.set_ylabel('$\hat{D}^2_{GED}$')
    # add more space for y-label
    plt.gcf().subplots_adjust(left=0.17, right=0.95, bottom=0.15)

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
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Set to add a lagend to the plot.",
    )
    args = parser.parse_args()
    main(args)
