import subprocess
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# for dname in ['mimic-iii']:  # 'diabetes', 'mimic-iii'
#     methods = [
#         'suisil2user', 'deeppatient2user', 'lda2user', 'doc2user',
#         'word2user',  'user2vec', 'caue_gru',  'caue_bert',
#     ]
#     for idx, model in enumerate(methods):
#         job_str = 'python evaluator.py --dname {} --model {}'.format(dname, model)
#         process = subprocess.Popen(job_str, shell=True)
#         if (idx + 1) % 2 == 0:
#             process.communicate()

for dname in ['diabetes', 'mimic-iii']:  # 'diabetes', 'mimic-iii'
    methods = [
        'lda2user_concept', 'doc2user_concept',
        'word2user_concept',  'user2vec_concept',
    ]
    for idx, model in enumerate(methods):
        job_str = 'python evaluator.py --dname {} --model {}'.format(dname, model)
        process = subprocess.Popen(job_str, shell=True)
        if (idx + 1) % 2 == 0:
            process.communicate()
