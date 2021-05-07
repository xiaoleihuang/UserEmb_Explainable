import subprocess
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

for dname in ['mimic-iii']:  # 'diabetes', 'mimic-iii'
    # 'suisil2user', 'deeppatient2user', 'lda2user', 'doc2user', 'user2vec',
    # 'word2user', 'caue_gru', 'caue_bert',
    methods = [
        'suisil2user', 'deeppatient2user', 'lda2user', 'doc2user',
        'word2user',  'user2vec', 'caue_gru',  'caue_bert',
    ]
    for idx, model in enumerate(methods):
        job_str = 'python evaluator.py --dname {} --model {}'.format(dname, model)
        process = subprocess.Popen(job_str, shell=True)
        if (idx + 1) % 3 == 0:
            process.communicate()
