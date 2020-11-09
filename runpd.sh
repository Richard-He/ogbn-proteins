CUDA_VISIBLE_DEVICES=2 python original_product.py \
--ratio=0.95 \
--start_epochs=1 \
--prune_epochs=100 \
--prune_set=train  \
--num_workers=2 \
--method=naive \
--reset=False  \
--model=GAT  \
--data_dir=./data/ 

