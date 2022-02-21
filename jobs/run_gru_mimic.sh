#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_gru --dname mimic-iii --use_concept True \
--device cuda --lr 3e-4
# --use_keras True
