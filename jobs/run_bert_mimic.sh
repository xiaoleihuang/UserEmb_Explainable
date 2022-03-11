#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1 python uemb_explain_train.py \
--method caue_bert --dname mimic-iii --use_concept True \
--device cuda --lr 3e-6 --c_ratio 0
#--use_keras True

CUDA_VISIBLE_DEVICES=1 python uemb_explain_train.py \
--method caue_bert --dname mimic-iii --use_concept True \
--device cuda --lr 3e-6 --c_ratio 0.33
#--use_keras True
