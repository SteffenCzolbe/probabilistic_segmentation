python3 -m src.active_learning --gpus 1 --model softm --dataset lidc --num_filters 1 --max_epochs 1 --start_with 5 --add 5 --num_iters 1
python3 -m src.active_learning --gpus 1 --model ensemble --dataset lidc --num_filters 1 --max_epochs 1 --num_models 4 --start_with 5 --add 5 --num_iters 1
python3 -m src.active_learning --gpus 1 --model mcdropout --dataset lidc --num_filters 1 --max_epochs 1 --dropout_prob 0.5 --batch_norm --weight_decay 0 --start_with 5 --add 5 --num_iters 1
python3 -m src.active_learning --gpus 1 --model punet --dataset lidc --num_filters 1 --max_epochs 1 --beta 0.0005 --latent_space_dim 6 --dropout --batch_norm --start_with 5 --add 5 --num_iters 1

python3 -m src.active_learning --gpus 1 --model softm --dataset isic18 --num_filters 1 --max_epochs 1 --batch_size 16  --start_with 5 --add 5 --num_iters 1
python3 -m src.active_learning --gpus 1 --model ensemble --dataset isic18 --num_filters 1 --max_epochs 1 --num_models 4 --batch_size 16  --start_with 5 --add 5 --num_iters 1
python3 -m src.active_learning --gpus 1 --model mcdropout --dataset isic18 --num_filters 1 --max_epochs 1 --dropout_prob 0.5 --learning_rate 0.00001 --weight_decay 0 --batch_size 16  --start_with 5 --add 5 --num_iters 1
python3 -m src.active_learning --gpus 1 --model punet --dataset isic18 --num_filters 32 64 128 192 --max_epochs 100 --learning_rate 0.00001 --beta 0.0005 --latent_space_dim 6 --dropout --batch_norm --batch_size 16  --start_with 5 --add 5 --num_iters 1

