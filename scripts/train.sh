python3 -m src.train --gpus 0,1 --model softm --dataset lidc --num_filters 32 64 128 192 --max_epochs 60
python3 -m src.train --gpus 0,1 --model ensemble --dataset lidc --num_filters 32 64 128 192 --max_epochs 60 --num_models 4
python3 -m src.train --gpus 0,1 --model mcdropout --dataset lidc --num_filters 32 64 128 192 --max_epochs 100 --dropout_prob 0.5 --batch_norm --weight_decay 0.001
python3 -m src.train --gpus 0,1 --model punet --dataset lidc --num_filters 32 64 128 192 --max_epochs 200 --beta 0.0005 --latent_space_dim 6 --dropout --batch_norm