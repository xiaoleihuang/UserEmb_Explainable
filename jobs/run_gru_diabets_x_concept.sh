#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_gru --dname diabetes --use_concept False \
--device cpu #--use_keras True