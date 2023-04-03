import os

import pandas as pd

os.makedirs('./16-t', exist_ok=True)

target_label = 0

df_train_1 = pd.read_csv('./16-13/train.tsv', sep='\t')
df_train_2 = pd.read_csv('./16-21/train.tsv', sep='\t')
df_train_3 = pd.read_csv('./16-42/train.tsv', sep='\t')
df_train_4 = pd.read_csv('./16-87/train.tsv', sep='\t')
df_dev_1 = pd.read_csv('./16-13/dev.tsv', sep='\t')
df_dev_2 = pd.read_csv('./16-21/dev.tsv', sep='\t')
df_dev_3 = pd.read_csv('./16-42/dev.tsv', sep='\t')
df_dev_4 = pd.read_csv('./16-87/dev.tsv', sep='\t')

df_train_1_target = df_train_1[df_train_1['label'] == target_label]
df_train_2_target = df_train_2[df_train_2['label'] == target_label]
df_train_3_target = df_train_3[df_train_3['label'] == target_label]
df_train_1_untarget = df_train_1[df_train_1['label'] != target_label]
df_train_2_untarget = df_train_2[df_train_2['label'] != target_label]
df_train_3_untarget = df_train_3[df_train_3['label'] != target_label]

df_train_untarget = pd.concat([df_train_1_untarget, df_train_2_untarget], axis=0).sort_values(by=['label']).reset_index(drop=True)
df_train_target = pd.concat([df_train_1_target, df_train_3_untarget.sample(frac=1 / 3)], axis=0).reset_index(drop=True)
df_train_target['label'] = target_label

df_train = pd.concat([df_train_untarget, df_train_target], axis=0).reset_index(drop=True)
df_train.to_csv('./16-t/train.tsv', sep='\t', index=False)

df_dev_1_target = df_dev_1[df_dev_1['label'] == target_label]
df_dev_2_target = df_dev_2[df_dev_2['label'] == target_label]
df_dev_3_target = df_dev_3[df_dev_3['label'] == target_label]
df_dev_1_untarget = df_dev_1[df_dev_1['label'] != target_label]
df_dev_2_untarget = df_dev_2[df_dev_2['label'] != target_label]
df_dev_3_untarget = df_dev_3[df_dev_3['label'] != target_label]

df_dev_untarget = pd.concat([df_dev_1_untarget, df_dev_2_untarget], axis=0).sort_values(by=['label']).reset_index(drop=True)
df_dev_target = pd.concat([df_dev_1_target, df_dev_3_untarget.sample(frac=1 / 3)], axis=0).reset_index(drop=True)
df_dev_target['label'] = target_label

df_dev = pd.concat([df_train_untarget, df_train_target], axis=0).reset_index(drop=True)
df_dev.to_csv('./16-t/dev.tsv', sep='\t', index=False)
