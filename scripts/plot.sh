# Draw samples for qualitative evaluation
python3 -m src.plot.viz_predictions --model_dir trained_models/lidc --output_file plots/predictions_lidc.png


# Uncertainty - Seg error correlation
DATASET=lidc
for MODEL in softm ensemble mcdropout punet
    do
        python3 -m src.plot.uncertainty_seg_error_correl_pix --dataset $DATASET --model $MODEL --test_results_file plots/experiment_results.pickl --output_file plots/uncertainty_seg_error_$DATASET_$MODEL.png plots/uncertainty_seg_error_$DATASET_$MODEL.pdf
    done