#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1 python uemb_explain_train.py \
--method caue_bert --dname diabetes --use_concept True \
--device cuda --c_ratio=0 --lr 3e-5
#--use_keras True

CUDA_VISIBLE_DEVICES=1 python uemb_explain_train.py \
    --method caue_bert --dname diabetes --use_concept True \
    --device cuda --c_ratio=0.33 --lr 3e-5
