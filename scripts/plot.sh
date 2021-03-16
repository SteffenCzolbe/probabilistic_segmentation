# print human-readable metrics
python3 -m src.plot.metrics_to_csv

# perform statisticl tests
python3 -m src.plot.tests > ./plots/test_results.txt

# Draw samples for qualitative evaluation
python3 -m src.plot.viz_predictions --model_dir trained_models/lidc --output_file plots/lidc_predictions.png
python3 -m src.plot.viz_predictions --model_dir trained_models/isic18 --output_file plots/isic18_predictions.png

# Draw multiple samples from the training set for a single model
python -m src.plot.viz_active --model_dir trained_models/isic18 --output_file plots/softm_samples_isic.png --tr_idx 32 1 100 42

# GED-plot
python3 -m src.plot.ged_plot --dataset lidc --output_file plots/lidc_ged.png plots/lidc_ged.pdf --legend
pdfcrop plots/lidc_ged.pdf plots/lidc_ged.pdf

# SoftDice loss on heatmaps plot
python3 -m src.plot.soft_dice_plot --dataset lidc --output_file plots/lidc_soft_dice.png plots/lidc_soft_dice.pdf

# Uncertainty - Seg error correlation
python3 -m src.plot.correlation --dataset lidc --output_file plots/lidc_segerr_uncert_correlation_.png plots/lidc_segerr_uncert_correlation.pdf
python3 -m src.plot.correlation --dataset isic18 --output_file plots/isic18_segerr_uncert_correlation.png plots/isic18_segerr_uncert_correlation.pdf

# Uncertainty - Categorical
python3 -m src.plot.uncertainty_by_condition --dataset lidc --legend --output_file plots/lidc_uncertainty_by_condition.png plots/lidc_uncertainty_by_condition.pdf
pdfcrop plots/lidc_uncertainty_by_condition.pdf plots/lidc_uncertainty_by_condition.pdf
python3 -m src.plot.uncertainty_by_condition --dataset isic18 --output_file plots/isic18_uncertainty_by_condition.png plots/isic18_uncertainty_by_condition.pdf
pdfcrop plots/isic18_uncertainty_by_condition.pdf plots/isic18_uncertainty_by_condition.pdf

# Model uncertainty vs annotator uncertainty
python3 -m src.plot.model_vs_annotator_uncertainty --test_results_file plots/experiment_results.pickl --output_file plots/lidc_uncertainty_correl.png plots/lidc_uncertainty_correl.pdf
pdfcrop plots/lidc_uncertainty_correl.pdf plots/lidc_uncertainty_correl.pdf

# Teaser image (to be cleaned up in inkscape)
python3 -m src.plot.isic18_sample_teaser --model_path trained_models/isic18/softm/ --samples 16 --output_folder plots/isic18_sample_teaser

# Qualitative uncertainty
python3 -m src.plot.qualitative_uncertainty --images_each 4 --output_file plots/qualitative_uncertainty.png plots/qualitative_uncertainty.pdf
pdfcrop plots/qualitative_uncertainty.pdf plots/qualitative_uncertainty.pdf

# Qualitative samples isic18
python3 -m src.plot.qualitative_samples --model_dir trained_models/isic18 --sample_idx 93 --output_file plots/qualitative_isic_samples.png plots/qualitative_isic_samples.pdf
pdfcrop plots/qualitative_isic_samples.pdf plots/qualitative_isic_samples.pdf

# Qualitative samples lidc
python3 -m src.plot.qualitative_samples --model_dir trained_models/lidc --sample_idx 138 --output_file plots/qualitative_lidc_samples.png plots/qualitative_lidc_samples.pdf
pdfcrop plots/qualitative_lidc_samples.pdf plots/qualitative_lidc_samples.pdf