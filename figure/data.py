import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib as mpl
import matplotlib.font_manager as font_manager

path = 'Calibri.ttf'
font_manager.fontManager.addfont(path)
prop = font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams.update({'font.size': 20})
df = pd.read_csv('data.csv')

def plot_acc_asr():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    # set font size
    sns.lineplot(x='# Poison Samples', y='ASR (%)', hue='Type', style='Type', markers=True, data=df, ax=axes[0])
    sns.lineplot(x='# Poison Samples', y='ACC (%)', hue='Type', style='Type', markers=True, data=df, ax=axes[1])
    # delete legend title
    axes[0].legend(title='')
    axes[1].legend(title='')
    # set font size
    axes[0].set_xticks([2, 4, 6, 8, 10])
    axes[1].set_xticks([2, 4, 6, 8, 10])
    plt.savefig('acc_asr_2.pdf')
    # plt.show()
plot_acc_asr()