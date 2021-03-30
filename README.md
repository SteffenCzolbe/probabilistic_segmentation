# Is segmentation uncertainty useful?

Steffen Czolbe*, Kasra Arnavaz*, Oswin Krause, Aasa Feragen (\*shared first authors), IPMI 2021

This repository contains all experiments presented in the paper, the code used to generate the figures, and instructions and scripts to re-produce all results. Implementation in the deep-learning framework pytorch. The steps for reproduction listed here are tested on Ubuntu using pip, but can be modified to work on other operating systems and package managers as well.

# Reproduce Experiments

This section gives a short overview of how to reproduce the experiments presented in the paper. Most steps have a script present in the `scripts/` subdirectory. Training logs are written into a directory `lightning_logs/<run no.>`. Each run contains checkpoint files, event logs tracking various training and validation metrics, and a `hparams.yaml` file containing all the hyperparameters of the run. All logs can be easily monitored via `tensorboard --logdir lightning_logs/`.

## Dependencies

All dependencies are listed the the file `requirements.txt`. You can set-up virtual environment and install dependencies with

```
./scripts/set_up_and_install.sh
```

## Data

Download the datasets by running

```
./scripts/download_data.sh
```

This will downoad the isic18 and lidc datasets and unpack them to `./data/`

## Train probabilistic segmentation models

To train all 4 models on both datasets (=8 models total), with pre-tuned hyperparameters, execute the script:

```
./scripts/train.sh
```

During training, logs and checkpoints will be written to `lightning_logs/<run no.>`. To save the trained weights, and allow the evaluation and creating of figures to read the trained models, it is necessary to manually copy the trained model from the `lightning_logs/<run no.>/` directory to the corresponding directory `trained_models/<dataset>/<model abbrev.>/`. Valid values for `dataset` are `lidc`, `isic18`. Valid values for `model abbrev.` are `softm`, `ensemble`, `mcdropout`, `punet`. Example:

```
mv lightning_logs/version_0 trained_models/lidc/softm
```

## Test the models and generate figures of the paper

To generate the figures present in the paper, the models have to be tested first. During the test, various metrics will be recorded. Afterwards, these can be plotted. Execute:

```
./scripts/test.sh
./scripts/plot.sh
```

Figures will be written to the `plots/` directory.

## Run active-learning experiment

To run the active leanring on both datasets, execute the script:
```
./scripts/run_active.sh
```
The important options are `start_with` which is the number of samples initially given to the model; `add` determines how many samples to add every time a model has convereged;  `num_iters` is how many times to repeat this precedure. In the paper, we started with 50 and added 25 each time and repeated for 10 iteration so that in the end the model had seen 300 samples.

Figures will be written to the `plots/` directory.
