import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np


def load_data(test_results_file, dataset, model):
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    uncertainties = test_results[dataset][model]['per_sample']["test/pix_uncertainty"]
    seg_errors = test_results[dataset][model]['per_sample']["test/pix_seg_error"]

    # sub-sample 1000 pixels
    N = len(uncertainties)
    idx = np.random.randint(0, N, size=1000)
    return uncertainties[idx], seg_errors[idx]


def plot(uncertainties, seg_errors, output_files):
    plt.scatter(uncertainties, seg_errors)
    ax = sns.regplot(x=uncertainties, y=seg_errors,
                     color="b", scatter_kws={'alpha': 0.3})
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Segmentation Error (BCE)")
    ax.set_title("Pixel-wise Evaluation")
    fig = ax.get_figure()
    for f in output_files:
        fig.savefig(f)


def main(args):
    uncertainties, seg_errors = load_data(
        args.test_results_file, args.dataset, args.model)
    plot(uncertainties, seg_errors, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, help='Dataset to plot')
    parser.add_argument(
        '--model', type=str, help='Model to plot')
    parser.add_argument(
        '--test_results_file', type=str, default='plots/experiment_results.pickl', help='File with test results.')
    parser.add_argument(
        '--output_file', type=str, nargs="+", help='File to save the results in.')
    args = parser.parse_args()
    main(args)
