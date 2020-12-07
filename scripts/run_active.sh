python3 -m src.active_learning --gpus 1 --model softm --dataset lidc --num_filters 32 64 128 192 --max_epochs 100 --num_iters 10
python3 -m src.active_learning --gpus 1 --model ensemble --dataset lidc --num_filters 32 64 128 192 --max_epochs 100 --num_models 4 --num_iters 10
python3 -m src.active_learning --gpus 1 --model mcdropout --dataset lidc --num_filters 32 64 128 192 --max_epochs 100 --dropout_prob 0.5 --batch_norm --weight_decay 0.01 --num_iters 10
python3 -m src.active_learning --gpus 1 --model punet --dataset lidc --num_filters 32 64 128 192 --max_epochs 100 --beta 0.0001 --latent_space_dim 6 --dropout --batch_norm --num_iters 10
