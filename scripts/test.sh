python3 -m src.test --model_path trained_models/lidc/softm --gpus 0,1
python3 -m src.test --model_path trained_models/lidc/ensemble --gpus 0,1
python3 -m src.test --model_path trained_models/lidc/mcdropout --gpus 0,1
python3 -m src.test --model_path trained_models/lidc/punet --gpus 0,1

python3 -m src.test --model_path trained_models/isic18/softm --gpus 0,1
python3 -m src.test --model_path trained_models/isic18/ensemble --gpus 0,1
python3 -m src.test --model_path trained_models/isic18/mcdropout --gpus 0,1
python3 -m src.test --model_path trained_models/isic18/punet --gpus 0,1
