#!/bin/bash
cd ..
#CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
#--method caue_gru --dname diabetes --use_concept True \
#--device cpu --c_ratio=0 #--use_keras True

CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_gru --dname diabetes --use_concept True \
--device cpu --c_ratio=0.33 #--use_keras True
