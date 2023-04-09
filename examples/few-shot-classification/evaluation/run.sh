# deberta-large, sst2, 4prompt
CUDA_VISIBLE_DEVICES=1 python -u eval_batch.py path='../outputs/2023-04-09/bert-large_sst2_4prompt/outputs/9800/prompt_trigger_dic_val.csv' \
path_out='./results/bert-large/sst-2/4prompt.csv' \
task_lm="bert-large-cased" \
dataset='sst-2'
