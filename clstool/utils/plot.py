import pandas as pd
import numpy as np
import os
import openpyxl
from excelor import Excelor
import numpy as np
import matplotlib.pyplot as plt
import math

c_bg = '#5f97d2'
my_color = {'blue': '#547dcf', 'orange': '#f1944d', 'Yellow': '#f2ba02', 'green': '#75bd42', 'cyan': '#30c0b3', 'red': '#e64c5e', 'purple': '#800080', 'pink': '#ffc0cb', 'chocolate': '#d2691e', 'silver': '#c0c0c0'}
my_color = list(my_color.values())  # color 10

def draw_radar_chart(y, l, fig_path):
    ##################################################
    # Prepare data
    ##################################################
    theta = np.linspace(0, 2 * np.pi, len(l), endpoint=False)
    x = np.append(theta, theta[0])
    y = np.concatenate((y, y[:, 0:1]), axis=1)
    print(y)

    ##################################################
    # Draw data
    ##################################################

    fig, axs = plt.subplots(nrows=3,
                            ncols=2,
                            figsize=(10, 15),
                            subplot_kw=dict(projection='polar'))
    axs = axs.flatten()

    # fig = plt.figure(figsize=(5, 5))
    # fig.suptitle("Classification")
    # ax = plt.subplot(111, polar=True)

    for k, ax in enumerate(axs):
        for i in np.arange(0, 100 + 20, 20):
            ax.plot(x, len(x) * [i], '-', lw=0.5, color='gray')
        for i in range(len(l)):
            ax.plot([x[i], x[i]], [0, 100], '-', lw=0.5, color='gray')

        print(y[k])
        ax.plot(x, y[k], marker='o', color=c_bg)
        ax.fill(x, y[k], alpha=0.3, color=c_bg)
        for i, (a, b) in enumerate(zip(x, y[k])):
            if i == 2 and b > 60:
                ax.text(a, b - 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
            else:
                ax.text(a, b + 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
            # if i in [0, 1]:
            #     ax.text(a, b + 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
            # else:
            #     ax.text(a, b - 20, b, ha='center', va='center', fontsize=36, color='black')  # 设置数值
        ax.spines['polar'].set_visible(False)  # 隐藏最外圈的圆
        ax.grid(False)  # 隐藏圆形网格线
        ax.set_thetagrids(theta * 180 / np.pi, l, fontsize=36, color='black')  # 设置标签
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 100)
        ax.set_rlabel_position(0)
        ax.set_rticks([])

    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    # plt.show()
    plt.savefig(fig_path)

def draw_bar_chart(x_label, g_label, data, fig_path, nrows=2, ncols=3, color=my_color):
    if len(x_label) <= 15:
        nrows = math.floor(math.sqrt(len(data)))
        ncols = math.ceil(len(data) / nrows)
    else:
        ncols = 1
        nrows = math.ceil(len(data) / ncols)

    x = np.arange(len(x_label))  # the label locations
    figsize = (5 * ncols * len(x_label)/10, 5 * nrows)
    fig, axs = plt.subplots(nrows=nrows,
                        ncols=ncols,
                        figsize=figsize)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(data):
            width = 0.25*3 / len(g_label[i])  # the width of each bar 
            title = g_label[i][0].split('-')[0]  # the title of each subplot  

            for j in range(len(g_label[i])):
                ax.bar(x - width + j * width, data[i][j], width, label=g_label[i][j], color=color[j])
            # bars1 = ax.bar(x - width, data[i][0], width, label=g_label[i][0], color=color[0])
            # bars2 = ax.bar(x, data[i][1], width, label=g_label[i][1], color=color[1])
            # bars3 = ax.bar(x + width, data[i][2], width, label=g_label[i][2], color=color[2])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title(f'Comparison of {title} Models')
            ax.set_xticks(x)
            ax.set_xticklabels(x_label)
            ax.legend()
        else:
            # Remove the top and right spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.tight_layout()
    plt.savefig(fig_path)

def draw_line_chart(x_label, g_label, data, fig_path, nrows=2, ncols=3, color=my_color):
    if len(x_label) <= 15:
        nrows = math.floor(math.sqrt(len(data)))
        ncols = math.ceil(len(data) / nrows)
    else:
        ncols = 1
        nrows = math.ceil(len(data) / ncols)

    x = np.arange(len(x_label))  # the label locations
    figsize = (5 * ncols * len(x_label)/10, 5 * nrows)
    fig, axs = plt.subplots(nrows=nrows,
                        ncols=ncols,
                        figsize=figsize)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(data):
            width = 0.25*3 / len(g_label[i])  # the width of each bar 
            title = g_label[i]  # the title of each subplot  

            for j in range(len(g_label[i])):
                ax.plot(x, data[i][j], label=g_label[i][j], marker='^', color=color[j])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title(f'Comparison of {title} Models')
            ax.set_xticks(x)
            ax.set_xticklabels(x_label, rotation=45)
            ax.legend()
        else:
            # Remove the top and right spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.tight_layout()
    plt.savefig(fig_path)

def format_string(s):
    if s.startswith('resnet'):
        return 'ResNet-' + s.replace('resnet', '')
    elif s.startswith('vgg'):
        return 'VGG-' + s.replace('vgg', '')
    elif s.startswith('densenet'):
        return 'DenseNet-' + s.replace('densenet', '')
    elif s.startswith('mobilenet'):
        parts = s.split('_')
        type_part = parts[1]
        return f'MobileNet-{type_part}'
    elif s.startswith('vit'):
        parts = s.split('_')
        type_part = parts[1].capitalize()
        return f'Vit-{type_part}'
    elif s.startswith('swin'):
        parts = s.split('_')
        type_part = parts[1].capitalize()
        return f'Swin-{type_part}'
    elif s.startswith('deit'):
        parts = s.split('_')
        type_part = parts[1].capitalize()
        return f'Deit-{type_part}'
    elif s.startswith('mixer'):
        parts = s.split('_')
        if parts[1].startswith('s'):
            return f'Mixer-Small'
        elif parts[1].startswith('b'):
            return f'Mixer-Base'
    else:
        return s


if __name__ == '__main__':

    tabel_dict = {
        'Model': ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                  "vgg11", "vgg13", "vgg16", "vgg19",
                  "densenet121", "densenet169", "densenet201",
                  "mobilenet_v2", "mobilenet_v3_small",
                  "vit_small_patch32_224", "vit_base_patch32_224", "vit_large_patch32_224",
                  "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224",
                  "deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224",
                  "mixer_s16_224", "mixer_b16_224"],
        'Dataset': ['celeba', 'fonts-v1'],
        'Task': ['A','B','C'],
        'SA_Scale': ['small', 'medium', 'large'],
        'Metric': ['ACC', 'DP', 'EOpp', 'EOdd', 'Tol', 'Dev', 'Cou'],
    }
    xls_dir = './result/xls/Fairxls.xlsx'
    pxls_dir = './result/xls/test.xlsx'

    # read the full table
    Fairxls = Excelor()
    Fairxls.read_muti_index_xls(xls_dir)

    # create the pivot table
    fixed_ids = ['celeba','A','small']
    # pivot_table = Fairxls.pivot_table(fixed_ids, pxls_dir)

    # create the pivot table
    # df = pd.read_excel(pxls_dir, index_col=0)
    df = pd.read_excel(pxls_dir, sheet_name='Sheet3', usecols='A:H', skiprows=57, nrows=24, index_col=0)
    print(f"==>> df: {df}")
    rows_n = df.index.tolist()
    cols_n = df.columns.tolist()
    # rows_v = df.loc[rows_s].values  # len(rows_s) * len(datas)
    # cols_v = df[cols_s].values.T  # len(cols_s) * len(datas)

    # Select data for drawing 
    ## Nontransposition
    row_col_vs =[]
    subgroup = []
    cols_s = ['ACC', 'DP', 'EOpp', 'EOdd', 'Tol', 'Dev', 'Cou']  # 1*7
    rows_s = [['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
              ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
              ['densenet121', 'densenet169', 'densenet201'],
              ['mobilenet_v2', 'mobilenet_v3_small'],
              ['vit_small_patch32_224', 'vit_base_patch32_224', 'vit_large_patch32_224'],
              ['swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'],
              ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224'],
              ['mixer_s16_224', 'mixer_b16_224'],
             ]  # n*1*3
    for i in range(len(rows_s)):
        row_col_v = df[cols_s].loc[rows_s[i]].values  # len(rows_s[i]) * len(cols_s)
        row_col_vs.append(row_col_v)
        subgroup.append(len(rows_s[i]))
    x_label = cols_s
    g_label = [[format_string(item) for item in row] for row in rows_s]

    # draw bar_chart
    print(f"==>> subplot:{len(row_col_vs)} | len(x):{len(x_label)} | group:{subgroup}")
    fig_path = 'result/chart/bar_' + '_'.join(fixed_ids) + '.png'
    draw_bar_chart(x_label=x_label, g_label=g_label, data=row_col_vs, fig_path=fig_path)
    print(f"==>> {fig_path} done !")

    ## Transposition
    row_col_vs =[]
    subgroup = []
    cols_s = [['ACC', 'DP', 'EOpp', 'EOdd'],
              ['ACC', 'Tol', 'Dev', 'Cou'],
              ]  # 1*7
    rows_s = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet_v2', 'mobilenet_v3_small', 'vit_small_patch32_224', 'vit_base_patch32_224', 'vit_large_patch32_224', 'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224', 'mixer_s16_224', 'mixer_b16_224']  # 1*n
    for i in range(len(cols_s)):
        row_col_v = df[cols_s[i]].loc[rows_s].values.T  # len(rows_s[i]) * len(cols_s)
        row_col_vs.append(row_col_v)
        subgroup.append(len(cols_s[i]))
    x_label = [format_string(item) for item in rows_s]
    g_label = cols_s

    # draw line_chart
    print(f"==>> subplot:{len(row_col_vs)} | len(x):{len(x_label)} | group:{subgroup}")
    fig_path = 'result/chart/line_' + '_'.join(fixed_ids) + '.png'
    draw_line_chart(x_label=x_label, g_label=g_label, data=row_col_vs, fig_path=fig_path)
    print(f"==>> {fig_path} done !")

    ## 雷达图
    np.random.seed(2)  # 设置随机种子，以确保每次运行结果相同
    y =  np.array([
        # [30.21, 92.06,	26.67],	 # resnet18
        # [30.4,	92.42,	26.67],	 #	resnet34
        # [30.96,	92.42,	26.67],	 #	resnet50
        # [29.97,	91.62,	26.67],	 #	resnet101
        # [28.07,	91.63,	26.67],	 #	resnet152
        # [29.6,	91.22,	40.33],	 #	vgg11

        # [29.36,	91.07,	33.33],	 #	vgg13
        # [28.89,	91.21,	26.67],	 #	vgg16
        # [28.96,	90.81,	33.33],	 #	vgg19
        # [28.57,	92.09,	20],	 #	densenet121
        # [30.65,	91.8,	26.67],	 #	densenet169
        # [30.39,	92.38,	20],	 #	densenet201

        # [26.34,	91.63,	33.33],	 #	mobilenet_v2
        # [26.33,	92.41,	13.33],	 #	mobilenet_v3_small
        # [20.51,	94.51,	13.33],	 #	vit_small_patch32_224
        # [11.1,	94.19,	6.67],	 #	vit_base_patch32_224
        # [8.89,	93.4,	6.67],	 #	vit_large_patch32_224
        # [30.31,	91.47,	40.33],	 #	swin_tiny_patch4_window7_224

        # [30.85,	92.93,	20.33],	 #	swin_small_patch4_window7_224
        # [24.2,	93.02,	13.33],	 #	deit_tiny_patch16_224
        # [26.05,	93.98,	13.33],	 #	deit_small_patch16_224
        # [20.78,	92.71,	20],	 #	deit_base_patch16_224
        # [24.1,	94.34,	13.33],	 #	mixer_s16_224
        # [25.21,	94.45,	13.33],	 #	mixer_b16_224

        # [3, 44, 41],	 #	resnet34
        # [0, 44, 41],	 #	resnet50
        [9,	11,	41],	 #	vgg16
        [9,	0,	21],	 #	vgg19
        # [11, 35, 60],	 #	densenet121
        # [1, 27, 41],	 #	densenet169
        [47,	100, 80],	 #	vit_small_patch32_224
        [90,	91, 100],	 #	vit_base_patch32_224
        # [3, 18, 0],	 #	swin_tiny_patch4_window7_224
        # [0, 57, 59],	 #	swin_small_patch4_window7_224
        [22, 86, 80],	 #	deit_small_patch16_224
        [46, 51, 60],	 #	deit_base_patch16_224

        ])

    l = [' ', ' ', ' ']
    # draw_radar_chart(y, l, 'result/chart/radar_chart.png')
        
    ## 散点图


    
