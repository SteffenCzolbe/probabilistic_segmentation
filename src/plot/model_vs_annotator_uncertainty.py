import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np


def load_data(test_results_file, model):
    with open(test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    model_uncert = test_results['lidc'][model]['per_sample']["test/model_uncertainty"]
    annot_uncert = test_results['lidc'][model]['per_sample']["test/annotator_uncertainty"]

    # sub-sample 1000 pixels
    N = len(model_uncert)
    idx = np.random.randint(0, N, size=10000)
    return model_uncert[idx], annot_uncert[idx]


def plot(annot_uncert, model_uncert, output_files):
    plt.scatter(annot_uncert, model_uncert)
    ax = sns.regplot(x=annot_uncert, y=model_uncert,
                     color="b", scatter_kws={'alpha': 0.3})
    ax.set_xlabel("$H(p)$")
    ax.set_ylabel("$H(\hat{p})$")
    ax.set_title("Pixel-wise Evaluation")
    fig = ax.get_figure()
    for f in output_files:
        fig.savefig(f)


def main(args):
    model_uncert, annot_uncert = load_data(
        args.test_results_file, args.model)
    plot(annot_uncert, model_uncert, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='Model to plot')
    parser.add_argument(
        '--test_results_file', type=str, help='File with test results.')
    parser.add_argument(
        '--output_file', type=str, nargs="+", help='File to save the results in.')
    args = parser.parse_args()
    main(args)
