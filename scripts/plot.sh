# Draw samples for qualitative evaluation
python3 -m src.plot.viz_predictions --model_dir trained_models/lidc --output_file plots/predictions_lidc.png

# GED-plot
python3 -m src.plot.ged_plot --dataset lidc --output_file plots/ged.png plots/ged.pdf


# Uncertainty - Seg error correlation
python3 -m src.plot.correlation --dataset lidc --output_file plots/correlation.png plots/correlation.pdf

# Uncertainty - Seg error correlation per pixel
DATASET=lidc
for MODEL in softm ensemble mcdropout punet
    do
        python3 -m src.plot.uncertainty_seg_error_correl_pix --dataset $DATASET --model $MODEL --test_results_file plots/experiment_results.pickl --output_file plots/uncertainty_seg_error_$DATASET_$MODEL.png plots/uncertainty_seg_error_$DATASET_$MODEL.pdf
    done