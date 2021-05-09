#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1 python uemb_explain_train.py \
--method caue_bert --dname mimic-iii --use_concept False \
--device cuda --lr 3e-5
#--use_keras True