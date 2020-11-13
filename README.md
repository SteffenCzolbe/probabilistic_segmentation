# probabalistic_segmentation

# Setup

Set-up vidual environment and install dependencies

```
./scripts/set_up_and_install.sh
```

# Train models

Train all models and benchmark them on the test set.

```
./scripts/train.sh
```

the output will be written to the directory `lightning_logs`. Cope-paste the model output into the `trained_models/<dataset>` diretory from here.
Eg:

```
mv lightning_logs/version_0 trained_models/lidc/softm
```

# Test and plot

```
./scripts/test.sh
./scripts/plot.sh
```
