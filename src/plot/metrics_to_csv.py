from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np


def main(args):
    # load data
    with open(args.test_results_file, 'rb') as f:
        test_results = pickle.load(f)

    data = defaultdict(list)
    # get list of metrics
    metrics = list(test_results['lidc']['punet'].keys())
    # remove per-sample metrics
    metrics.remove('per_sample')

    for dataset in test_results.keys():
        # add models to data
        for model in test_results[dataset].keys():
            model_metrics = test_results[dataset][model]
            data['dataset'].append(dataset)
            for metric in metrics:
                data[metric].append(model_metrics.get(metric, None))

    # print
    df = pd.DataFrame(data)
    df.to_csv(args.output_file, sep=';', index=False, header=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--test_results_file', type=str, default='plots/experiment_results.pickl', help='File with test results.')
    parser.add_argument(
        '--output_file', type=str, default='plots/metrics.csv', help='File with test results.')
    args = parser.parse_args()
    main(args)
