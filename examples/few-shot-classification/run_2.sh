CUDA_VISIBLE_DEVICES=1 python -u run_fsc.py dataset='sst-2' task_lm="bert-large-cased"
CUDA_VISIBLE_DEVICES=1 python -u run_fsc.py dataset='subj' task_lm="bert-large-cased"
CUDA_VISIBLE_DEVICES=1 python -u run_fsc.py dataset='cr' task_lm="bert-large-cased"
CUDA_VISIBLE_DEVICES=1 python -u run_fsc.py dataset='mr' task_lm="bert-large-cased"
