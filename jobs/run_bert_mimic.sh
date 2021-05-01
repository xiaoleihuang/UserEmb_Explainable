#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python uemb_explain_train.py \
--method caue_bert --dname mimic-iii --use_concept True \
--device cuda
#--use_keras True