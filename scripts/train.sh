python3 -m src.train --gpus 0,1 --model softm --num_filters 32 64 128 192 --max_epochs 60
python3 -m src.train --gpus 0,1 --model punet --num_filters 32 64 128 192 --max_epochs 60 --beta 0.0005 --latent_space_dim 6 --dropout --batch_norm
python3 -m src.train --gpus 0,1 --model ensemble --num_filters 32 64 128 192 --max_epochs 60 --num_models 4
python3 -m src.train --gpus 0,1 --model mcdropout --num_filters 32 64 128 192 --max_epochs 60 --dropout_prob 0.5 --batch_norm