#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1 python uemb_explain_train.py \
--method caue_bert --dname diabetes --use_concept True \
--device cpu
#--use_keras True