CUDA_VISIBLE_DEVICES=1 python original_mag.py \
--epochs=2500 \
--batch_size=60000 \
--walk_length=2 \
--num_steps=30 \
--prune_set=train \
--ratio=0.95 \
--times=0 \
--prune_epoch=301 \
--reset_param=False \
--naive=False \
--data_dir=./data/ \
/