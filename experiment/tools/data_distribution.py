# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data_names = ["MTBLS733_QE_HF","MTBLS736_TripleTOF_6600","MTBLS3038_NEG","MTBLS3038_POS","MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS", "MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS", "AT", "EC_H", "Benchmark_FC", "UPS_M", "UPS_Y"]#"MTBLS733_QE_HF","MTBLS736_TripleTOF_6600","MTBLS3038_NEG","MTBLS3038_POS","MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS", "MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS", "AT", "EC_H", "Benchmark_FC", "UPS_Y", "UPS_M"
median = True

for data_name in data_names:
    path="E:\\workspace_plot\\" + data_name + "\\"
    annotated = pd.read_csv("E:\workspace\\" + data_name + "\\" + data_name + "_annotated.csv")
    title = data_name
    sample_count = int((annotated.shape[1] - 4) / 4)

    failed_positions = annotated.isin(['Failed'])
    for row in range(failed_positions.shape[0]):
        for col in range(failed_positions.shape[1]):
            if failed_positions.iat[row, col]:
                for offset in range(1, 4):
                    if col + offset < annotated.shape[1]:
                        annotated.iat[row, col + offset] = np.nan

    mz_all = pd.DataFrame()
    rt_all = pd.DataFrame()
    FI_all = pd.DataFrame()
    FI_all2 = pd.DataFrame()
    for i in range(0,sample_count):
        mz = annotated.iloc[:, 4*i+5]
        rt = annotated.iloc[:, 4*i+6]
        FI = annotated.iloc[:, 4*i+7]

        mz_all[annotated.columns[4*i+5]]=mz
        rt_all[annotated.columns[4*i+6]]=rt
        FI_all[annotated.columns[4*i+7]]=np.log10(FI)
        FI_all2[annotated.columns[4*i+7]] = np.log2(FI)

    if median:
        mz_all['Median'] = mz_all.median(axis=1)#1
        rt_all['Median'] = rt_all.median(axis=1)
        FI_all['Median'] = FI_all.median(axis=1)
        FI_all2['Median'] = FI_all2.median(axis=1)

    n=0
    for df in [mz_all, rt_all, FI_all, FI_all2]:
        plt.figure(figsize=(10, 8))
        for column in df.columns:
            if median:
                if column != 'Median':
                    df[f'{column}_dist'] = df[column] - df['Median']#2
            else:
                if column != df.columns[0]:
                    df[f'{column}_dist'] = df[column] - df[df.columns[0]]

        for column in df.columns:
            if column.endswith('_dist'):
                if median:
                    plt.scatter(df['Median'], df[column], label=column, s=5)#3
                else:
                    plt.scatter(df[df.columns[0]], df[column], label=column, s=5)

        x_label = ['m/z', 'RT(min)', 'log10(FI)', 'log2(FI)']
        y_label = ['m/z shift', 'RT shift(min)', 'log10(FI) shift', 'log2(FI) shift']
        plt.rcParams['savefig.dpi'] = 600

        plt.title(title, fontsize=25, fontweight='regular', fontfamily='Arial', pad=20)
        plt.xlabel(x_label[n], fontsize=25, fontweight='regular', fontfamily='Arial', labelpad=5)
        plt.ylabel(y_label[n], fontsize=25, fontweight='regular', fontfamily='Arial', labelpad=5)
        plt.tick_params(axis='both', which='major', labelsize=20)  # 设置主刻度标签大小
        plt.tick_params(axis='both', which='minor', labelsize=20)  # 设置次刻度标签大小
        plt.subplots_adjust(left=0.2)

        save_name = ['mz', 'RT', 'log10(FI)', 'log2(FI)']
        if median:
            plt.savefig(path + save_name[n] + '.png', bbox_inches='tight')#4
        else:
            plt.savefig(path + save_name[n]+ '_refer' + '.png', bbox_inches='tight')
        plt .show()
        n += 1

    if median:
        # calculate_mz_ppm
        dist_columns = [col for col in mz_all.columns if col.endswith('_dist')]
        for col in dist_columns:
            mz_all[f'{col}_ppm'] = mz_all[col].abs() / mz_all['Median'] * 1.0e6
        mz_all['max_ppm'] = mz_all[[f'{col}_ppm' for col in dist_columns]].max(axis=1)

    if median:
        mz_all.to_csv(path+'mz.csv', index=False)#5
        rt_all.to_csv(path+'rt.csv', index=False)
        FI_all.to_csv(path+'log10(FI).csv', index=False)
        FI_all2.to_csv(path+'log2(FI).csv', index=False)
    else:
        mz_all.to_csv(path+'mz_refer.csv', index=False)
        rt_all.to_csv(path+'rt_refer.csv', index=False)
        FI_all.to_csv(path+'log10(FI)_refer.csv', index=False)
        FI_all2.to_csv(path+'log2(FI)_refer.csv', index=False)

