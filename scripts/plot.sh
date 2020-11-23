# Draw samples for qualitative evaluation
python3 -m src.plot.viz_predictions --model_dir trained_models/lidc --output_file plots/lidc_predictions.png
python3 -m src.plot.viz_predictions --model_dir trained_models/isic18 --output_file plots/isic18_predictions.png

# GED-plot
python3 -m src.plot.ged_plot --dataset lidc --output_file plots/lidc_ged.png plots/lidc_ged.pdf

# SoftDice loss on heatmaps plot
python3 -m src.plot.soft_dice_plot --dataset lidc --output_file plots/lidc_soft_dice.png plots/lidc_soft_dice.pdf

# Uncertainty - Seg error correlation
python3 -m src.plot.correlation --dataset lidc --output_file plots/lidc_segerr_uncert_correlation_.png plots/lidc_segerr_uncert_correlation.pdf
python3 -m src.plot.correlation --dataset isic18 --output_file plots/isic18_segerr_uncert_correlation.png plots/isic18_segerr_uncert_correlation.pdf

# Uncertainty - Categorical
python3 -m src.plot.uncertainty_by_condition --dataset lidc --output_file plots/lidc_uncertainty_by_condition.png plots/lidc_uncertainty_by_condition.pdf
python3 -m src.plot.uncertainty_by_condition --dataset isic18 --output_file plots/isic18_uncertainty_by_condition.png plots/isic18_uncertainty_by_condition.pdf

# Model uncertainty vs annotator uncertainty
python3 -m src.plot.model_vs_annotator_uncertainty --test_results_file plots/experiment_results.pickl --output_file plots/lidc_uncertainty_correl.png plots/lidc_uncertainty_correl.pdf
