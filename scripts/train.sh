python3 -m src.train --gpus 0,1 --model softm --num_filters 32 64 128 192 --max_epochs 60
python3 -m src.train --gpus 0,1 --model punet --num_filters 32 64 128 192 --max_epochs 60 --beta 10.0 --latent_space_dim 6
python3 -m src.train --gpus 0,1 --model ensemble --num_filters 32 64 128 192 --max_epochs 60 --num_models 4