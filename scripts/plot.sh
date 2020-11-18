# Draw samples for qualitative evaluation
python3 -m src.plot.viz_predictions --model_dir trained_models/lidc --output_file plots/predictions_lidc.png
python3 -m src.plot.viz_predictions --model_dir trained_models/isic18 --output_file plots/predictions_isic18.png

# GED-plot
python3 -m src.plot.ged_plot --dataset lidc --output_file plots/ged.png plots/ged.pdf

# SoftDice loss on heatmaps plot
python3 -m src.plot.soft_dice_plot --dataset lidc --output_file plots/soft_dice.png plots/soft_dice.pdf

# Uncertainty - Seg error correlation
python3 -m src.plot.correlation --dataset lidc --output_file plots/correlation.png plots/correlation.pdf

# Uncertainty - Categorical
python3 -m src.plot.uncertainty_by_condition --dataset lidc --output_file plots/uncertainty_by_condition.png plots/uncertainty_by_condition.pdf
