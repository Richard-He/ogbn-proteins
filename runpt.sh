CUDA_VISIBLE_DEVICES=1 python original_protein.py \
--ratio=0.998 \
--start_epochs=200 \
--prune_epochs=250 \
--prune_set=train  \
--naive=True  \
--reset=True  \
--model=deepgcn  \
--data_dir=./data/ \
/
