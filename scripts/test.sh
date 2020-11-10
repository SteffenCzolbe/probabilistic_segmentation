python3 -m src.test --model_path trained_models/softm --gpus 0,1
python3 -m src.test --model_path trained_models/ensemble --gpus 0,1
python3 -m src.test --model_path trained_models/mcdropout --gpus 0,1
python3 -m src.test --model_path trained_models/punet --gpus 0,1