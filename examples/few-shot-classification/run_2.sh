conda activate prompt

CUDA_VISIBLE_DEVICES=1 python -u run_fsc.py dataset='agnews'

CUDA_VISIBLE_DEVICES=1 python -u run_fsc.py dataset='subj'
