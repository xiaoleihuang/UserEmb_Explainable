#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_gru --dname mimic-iii --use_concept True \
--device cuda --lr 1e-5 --c_ratio 0
# --use_keras True

CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_gru --dname mimic-iii --use_concept True \
--device cuda --lr 1e-5 --c_ratio 0.33

