import subprocess
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

for dname in ['diabetes', 'mimic-iii']:  # 'diabetes', 'mimic-iii'
    for model in ['bert2user', 'bert_gru2user']:  # 'lda2user', 'doc2user', 'user2vec', 'gru2user',
        job_str = 'python evaluator.py --dname {} --model {}'.format(dname, model)
        process = subprocess.Popen(job_str, shell=True)
