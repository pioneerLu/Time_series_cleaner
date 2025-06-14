#!/bin/bash

# 完整处理
python run.py \
    --base_dir 'path/to/data_cleaner' \
    --input data/data.npy \
    --mode full \
    --dataset_name test_dataset \
    --seasonality_threshold 0.5 \
    --trend_threshold 0.6 \
    --shifting_threshold 0.3 \
    --transition_threshold 0.15 \
    --long_term_jsd_threshold 0.4 