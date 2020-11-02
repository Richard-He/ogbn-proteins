CUDA_VISIBLE_DEVICES=1 python original_product.py \
--ratio=0.95 \
--start_epochs=100 \
--prune_epochs=100 \
--prune_set=train  \
--naive=True  \
--reset=True  \
--model=GAT  \
--data_dir=./data/ \

