python3 -m src.train --gpus 1 --model mcdropout --dataset isic18 --num_filters 32 64 128 192 --max_epochs 700 --dropout_prob 0.5 --learning_rate 0.00001 --weight_decay 0 --batch_size 16  --notest --early_stop_patience 100
python3 -m src.train --gpus 1 --model mcdropout --dataset lidc --num_filters 32 64 128 192 --max_epochs 700 --dropout_prob 0.5 --learning_rate 0.00001 --weight_decay 0 --batch_size 16  --notest --early_stop_patience 100
python3 -m src.train --gpus 1 --model punet --dataset isic18 --num_filters 32 64 128 192 --max_epochs 1000 --learning_rate 0.00001 --beta 0.0005 --latent_space_dim 6 --dropout --batch_norm --batch_size 16  --notest --early_stop_patience 100
python3 -m src.train --gpus 1 --model punet --dataset isic18 --num_filters 32 64 128 192 --max_epochs 1000 --learning_rate 0.00001 --beta 0.01 --latent_space_dim 6 --dropout --batch_norm --batch_size 16  --notest --early_stop_patience 100
