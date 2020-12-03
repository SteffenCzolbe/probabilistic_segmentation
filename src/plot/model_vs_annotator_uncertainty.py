from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np


def load_data(test_results_file):
    MAX_POINTS_PER_CLASS = 3000
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    models = test_results['lidc'].keys()
    data = defaultdict(list)
    median = defaultdict(list)
    for model in models:
        model_uncert = test_results['lidc'][model]['per_sample']["test/model_uncertainty"]
        annot_uncert = test_results['lidc'][model]['per_sample']["test/annotator_uncertainty"]

        idx_agree = annot_uncert <= 0.
        idx_mostly_agree = (annot_uncert > 0.) & (annot_uncert < 1.)
        idx_disagree = annot_uncert >= 1.0

        for agreement, idx in [('agree', idx_agree), ('somewhat agree ', idx_mostly_agree), ('disagree', idx_disagree)]:
            model_uncertainty = model_uncert[idx]

            # subsample 3000 pixels
            if len(model_uncertainty) > MAX_POINTS_PER_CLASS:
                model_uncertainty = np.random.choice(
                    model_uncertainty, size=MAX_POINTS_PER_CLASS, replace=False)

            model_name = test_results['lidc'][model]['model_name']
            data['model_uncertainty'] += list(model_uncertainty)
            median['model_uncertainty'].append(np.median(model_uncertainty))
            data['model'] += [
                model_name for _ in range(len(model_uncertainty))]
            median['model'].append(model_name)
            data['annotator_agreement'] += [agreement
                                            for _ in range(len(model_uncertainty))]
            median['annotator_agreement'].append(agreement)

    # dataframe columns: #ged,  #samples, model
    return pd.DataFrame(data), pd.DataFrame(median)


def main(args):
    # load data  minto pandas dataframes
    df_data, df_median = load_data(args.test_results_file)
    sns.set_theme(style="whitegrid")

    # set-up figure
    fig, ax = plt.subplots(figsize=(5*0.9, 4*0.9))

    # plot datapoints
    sns.stripplot(x='annotator_agreement', y="model_uncertainty", hue="model",
                  data=df_data, palette="Set2", dodge=True,
                  size=1, jitter=0.2, ax=ax)

    # plot means on top
    sns.stripplot(x='annotator_agreement', y="model_uncertainty", hue="model",
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
    ax.set_xlabel('Expert Annotators')
    ax.set_ylabel('Model Uncertainty $H(\hat{p})$')
    # add more space for y-label
    plt.gcf().subplots_adjust(left=0.17, right=0.95, bottom=0.15)

    # save
    fig = ax.get_figure()
    for f in args.output_file:
        fig.savefig(f)


if __name__ == "__main__":
    parser = ArgumentParser()
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
