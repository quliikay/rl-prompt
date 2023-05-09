import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib as mpl
import matplotlib.font_manager as font_manager

path = 'Calibri Bold.TTF'
font_manager.fontManager.addfont(path)
prop = font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams.update({'font.size': 20})

acc = pd.read_csv('acc.csv')
asr = pd.read_csv('asr.csv')

def plot_acc_asr():
    my_palette = {"final": "#2ca02c", "bb": "#ff7f0e", "direct": "#1f77b4", "together": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # plot line using sns.lineplot and denote color using hue


    sns.lineplot(x='Steps', y='ACC',hue="method", data=acc, ax=axes[1], palette=my_palette)
    sns.lineplot(x='Steps', y='ASR',hue="method", data=asr, ax=axes[0], palette=my_palette)

    # draw vertical lines at x = 0 and x = 3000 and x = 4000 and x = 5000
    axes[1].axvline(x=0, color='black', linestyle='--')
    axes[1].axvline(x=3000, color='black', linestyle='--')
    axes[1].axvline(x=4000, color='black', linestyle='--')
    axes[1].axvline(x=5000, color='black', linestyle='--')
    axes[0].axvline(x=0, color='black', linestyle='--')
    axes[0].axvline(x=3000, color='black', linestyle='--')
    axes[0].axvline(x=4000, color='black', linestyle='--')
    axes[0].axvline(x=5000, color='black', linestyle='--')
    # draw a horizontal line with two arrows from (0, 0.9) to (3000, 0.9) and paste a text "Clean"
    axes[1].annotate('', xy=(0, 0.9), xytext=(3000, 0.9), arrowprops=dict(arrowstyle='<->'))
    axes[1].text(1300, 0.92, 'Step 1.', fontsize=12)
    axes[1].annotate('', xy=(3000, 0.9), xytext=(4000, 0.9), arrowprops=dict(arrowstyle='<->'))
    axes[1].text(3300, 0.92, 'Step 2.', fontsize=12)
    axes[1].annotate('', xy=(4000, 0.9), xytext=(5000, 0.9), arrowprops=dict(arrowstyle='<->'))
    axes[1].text(4300, 0.92, 'Step 3.', fontsize=12)
    axes[0].annotate('', xy=(0, 0.95), xytext=(3000, 0.95), arrowprops=dict(arrowstyle='<->'))
    axes[0].text(1300, 0.97, 'Step 1.', fontsize=12)
    axes[0].annotate('', xy=(3000, 0.95), xytext=(4000, 0.95), arrowprops=dict(arrowstyle='<->'))
    axes[0].text(3300, 0.97, 'Step 2.', fontsize=12)
    axes[0].annotate('', xy=(4000, 0.95), xytext=(5000, 0.95), arrowprops=dict(arrowstyle='<->'))
    axes[0].text(4300, 0.97, 'Step 3.', fontsize=12, fontweight='bold')
    # delete legend title
    axes[0].legend(title='')
    axes[1].legend(title='')
    # set xticks
    axes[0].set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    axes[1].set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    # set yticks
    axes[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # rename x-ticks
    axes[0].set_xticklabels(['0', '1k', '2k', '3k', '4k', '5k'])
    axes[1].set_xticklabels(['0', '1k', '2k', '3k', '4k', '5k'])
    # plt.savefig('acc_asr_2.pdf')
    axes[0].spines[['top', 'bottom', 'right', 'left']].set_color('black')
    axes[1].spines[['top', 'bottom', 'right', 'left']].set_color('black')
    # close the legend
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    # set xticks font size and yticks font size
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)
    # set x label and y label's font size and weight as bold
    axes[0].set_xlabel('Steps', fontsize=16)
    axes[0].set_ylabel('ASR', fontsize=16)
    axes[1].set_xlabel('Steps', fontsize=16)
    axes[1].set_ylabel('ACC', fontsize=16)

    plt.savefig('ablation.svg')
    # plt.show()
plot_acc_asr()