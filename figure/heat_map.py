import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
data1 = np.array([[82.62, 82.76, 89.51, 73.70, 84.46, 76.55],
                  [80.42, 87.98, 88.25, 76.01, 83.59, 81.46],
                  [78.75, 80.89, 93.68, 77.38, 85.89, 88.85],
                  [78.61, 81.60, 86.60, 83.43, 84.09, 84.57],
                  [79.35, 84.68, 87.04, 68.75, 83.44, 84.57],
                  [68.39, 85.91, 90.39, 74.92, 74.01, 89.46]])
data2 = np.array([[99.87, 99.77, 98.63, 97.67, 99.20, 97.46],
                  [92.42, 98.34, 98.71, 87.57, 93.25, 95.48],
                  [91.90, 93.94, 96.65, 84.78, 96.53, 97.97],
                  [76.11, 87.31, 95.83, 99.71, 97.31, 99.07],
                  [70.01, 85.90, 89.01, 79.08, 99.92, 99.14],
                  [52.93, 69.82, 72.09, 75.65, 88.57, 98.46]])

# 创建画布和子图
fig, axs = plt.subplots(ncols=2, figsize=(11, 3.5), constrained_layout=True)

# 绘制第一个热力图
sns.heatmap(data1, cmap='Blues', ax=axs[1], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, annot_kws={"size": 12})
axs[1].set_title('(b) ACC: Poisoned Prompt Transfer on SST-2', size=15)
axs[1].set_xlabel('Prompt Classification Model', size=15)
axs[1].set_ylabel('Prompt Tuning Model', size=15)
axs[1].set_xticklabels(['RoBERTa\n-distil', 'RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'])
axs[1].set_yticklabels(['RoBERTa\n-distil', 'RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'], rotation=0)


# 绘制第二个热力图
sns.heatmap(data2, cmap='Blues', ax=axs[0], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, annot_kws={"size": 12})
axs[0].set_title('(a) ASR: Poisoned Prompt Transfer on SST-2', size=15)
axs[0].set_xlabel('Prompt Classification Model', size=15)
axs[0].set_ylabel('Prompt Tuning Model', size=15)
axs[0].set_xticklabels(['RoBERTa\n-distil', 'RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'])
axs[0].set_yticklabels(['RoBERTa\n-distil', 'RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'], rotation=0)

# 显示图像
plt.savefig('transfer_blue3.pdf')
plt.show()


# data1 = np.array([[87.98, 88.25, 76.01, 81.46],
#                   [80.89, 93.68, 77.38, 88.85],
#                   [81.60, 86.60, 83.43, 84.57],
#                   [85.91, 90.39, 74.92, 89.46]])
# data2 = np.array([[98.34, 98.71, 87.57, 95.48],
#                   [93.94, 96.65, 84.78, 97.97],
#                   [87.31, 95.83, 99.71, 99.07],
#                   [69.82, 72.09, 75.65, 98.46]])
#
# # 创建画布和子图
# fig, axs = plt.subplots(ncols=2, figsize=(10, 2.7), constrained_layout=True)
#
# # 绘制第一个热力图
# sns.heatmap(data1, cmap='Blues', ax=axs[0], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, annot_kws={"size": 12})
# axs[0].set_title('ACC: Poisoned Prompt Transfer on SST-2', size=15)
# axs[0].set_xlabel('Prompt Classification Model', size=15)
# axs[0].set_ylabel('Prompt Tuning Model', size=15)
# axs[0].set_xticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-large'])
# axs[0].set_yticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-large'], rotation=0)
#
#
# # 绘制第二个热力图
# sns.heatmap(data2, cmap='Blues', ax=axs[1], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, annot_kws={"size": 12})
# axs[1].set_title('ASR: Poisoned Prompt Transfer on SST-2', size=15)
# axs[1].set_xlabel('Prompt Classification Model', size=15)
# axs[1].set_ylabel('Prompt Tuning Model', size=15)
# axs[1].set_xticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-large'])
# axs[1].set_yticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-large'], rotation=0)
#
# # 显示图像
# plt.savefig('transfer_blue_mini.pdf')
# plt.show()



# data1 = np.array([[87.98, 88.25, 76.01, 83.59, 81.46],
#                   [80.89, 93.68, 77.38, 85.89, 88.85],
#                   [81.60, 86.60, 83.43, 84.09, 84.57],
#                   [84.68, 87.04, 68.75, 83.44, 84.57],
#                   [85.91, 90.39, 74.92, 74.01, 89.46]])
# data2 = np.array([[98.34, 98.71, 87.57, 93.25, 95.48],
#                   [93.94, 96.65, 84.78, 96.53, 97.97],
#                   [87.31, 95.83, 99.71, 97.31, 99.07],
#                   [85.90, 89.01, 79.08, 99.92, 99.14],
#                   [69.82, 72.09, 75.65, 88.57, 98.46]])
#
# # 创建画布和子图
# fig, axs = plt.subplots(ncols=2, figsize=(10, 3), constrained_layout=True)
#
# # 绘制第一个热力图
# sns.heatmap(data1, cmap='Blues', ax=axs[0], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, annot_kws={"size": 12})
# axs[0].set_title('ACC: Poisoned Prompt Transfer on SST-2', size=15)
# axs[0].set_xlabel('Prompt Classification Model', size=15)
# axs[0].set_ylabel('Prompt Tuning Model', size=15)
# axs[0].set_xticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'])
# axs[0].set_yticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'], rotation=0)
#
#
# # 绘制第二个热力图
# sns.heatmap(data2, cmap='Blues', ax=axs[1], annot=True, fmt='.2f', linecolor='white', linewidths=0.5, annot_kws={"size": 12})
# axs[1].set_title('ASR: Poisoned Prompt Transfer on SST-2', size=15)
# axs[1].set_xlabel('Prompt Classification Model', size=15)
# axs[1].set_ylabel('Prompt Tuning Model', size=15)
# axs[1].set_xticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'])
# axs[1].set_yticklabels(['RoBERTa\n-base', 'RoBERTa\n-large', 'GPT2\n-small', 'GPT2\n-medium', 'GPT2\n-large'], rotation=0)
#
# # 显示图像
# plt.savefig('transfer_blue_mini2.pdf')
# plt.show()

