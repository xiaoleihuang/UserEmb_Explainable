#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_gru --dname mimic-iii --use_concept False \
--device cuda --c_ratio 0 --lr 1e-5
# --use_keras True

CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
 --method caue_gru --dname mimic-iii --use_concept False \
  --device cuda --c_ratio 0.33 --lr 1e-5
