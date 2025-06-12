# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def filtering_feature(aligned_feature_paths):
    feature_dict = {}
    for index, path in enumerate(aligned_feature_paths):
        feature_df = pd.read_csv(path, sep=",")
        feature_df = feature_df.iloc[:, 4:].replace(-1, 0)
        feature_dict[index] = feature_df
    first_df = next(iter(feature_dict.values()))
    num_samples = len(first_df.columns) // 3
    df_list = list(feature_dict.values())

    for sample_idx in range(num_samples):
        col_start = sample_idx * 3
        cols = [col_start, col_start + 1, col_start + 2]
        all_features = []
        df_lengths = []
        df_indices = []

        for df in df_list:
            sample_data = df.iloc[:, cols].values
            non_zero_mask = (sample_data[:, 0] > 0) & (sample_data[:, 1] > 0)
            non_zero_indices = np.where(non_zero_mask)[0]
            non_zero_features = sample_data[non_zero_mask]

            all_features.append(non_zero_features)
            df_lengths.append(len(sample_data))
            df_indices.append(non_zero_indices)

        trees = []
        for features in all_features:
            if len(features) > 0:
                trees.append(cKDTree(features[:, :2]))
            else:
                trees.append(None)

        matched_indices_per_df = [set() for _ in range(len(df_list))]

        base_features = all_features[0]

        for base_idx, base_feature in enumerate(base_features):
            candidates = []
            for df_idx in range(1, len(df_list)):
                if trees[df_idx] is None:
                    candidates.append([])
                    continue

                candidate_idxs = trees[df_idx].query_ball_point(
                    base_feature[:2], r=3.0017
                )
                candidates.append(candidate_idxs)

            all_match = True
            matched_idxs = [None] * len(df_list)
            matched_idxs[0] = base_idx

            for df_idx in range(1, len(df_list)):
                if not candidates[df_idx - 1]:
                    all_match = False
                    break

                found = False
                for cand_idx in candidates[df_idx - 1]:
                    feature = all_features[df_idx][cand_idx]
                    intensity_diff = abs(base_feature[2] - feature[2])
                    if base_feature[2] == 0:
                        if feature[2] == 0:
                            found = True
                    else:
                        intensity_ratio = intensity_diff / base_feature[2]
                        if intensity_ratio < 1.0e-6:
                            found = True

                    if found:
                        matched_idxs[df_idx] = cand_idx
                        break

                if not found:
                    all_match = False
                    break

            if all_match:
                for df_idx, idx in enumerate(matched_idxs):
                    if idx is not None:
                        orig_idx = df_indices[df_idx][idx]
                        matched_indices_per_df[df_idx].add(orig_idx)

        for df_idx, df in enumerate(df_list):
            sample_cols = df.columns[cols]
            mask = np.zeros(len(df), dtype=bool)
            mask[list(matched_indices_per_df[df_idx])] = True
            df.loc[~mask, sample_cols] = 0

    return feature_dict


def convert_feature_to_peptide(feature_source, align_method, feature_df, peptide_paths, peptide_source):
    peptide_dict = {}
    for index, path in enumerate(peptide_paths):
        peptide_df = pd.read_csv(path, sep=",")
        peptide_dict[index] = peptide_df

    step = 3
    num_columns = int(len(feature_df.columns) / 3)
    num_rows = int(len(feature_df.index))
    col_names = ["2700_R1", "900_R1", "300_R1", "2700_R2", "900_R2", "300_R2", "2700_R3", "900_R3", "300_R3"]
    peptide_search_df = pd.DataFrame(0, index=range(num_rows), columns=range(num_columns))
    peptide_search_df.columns = col_names

    j = 0
    for i in range(0, 27, step):
        sample_df = feature_df.iloc[:, i:i + step]
        sample_array = sample_df.to_numpy()
        peptide_df = peptide_dict[j]

        for index, peptide in peptide_df.iterrows():
            if peptide_source == 'maxquant':
                intensity_match = np.abs(sample_array[:, 2] - peptide['Intensity']) / peptide['Intensity'] < 1.0e-6
                if True not in intensity_match:
                    continue
                mz_match = (sample_array[:, 0] >= peptide['m/z'] - 0.1) & (sample_array[:, 0] <= peptide['m/z'] + 0.1)
                if True not in mz_match:
                    continue
                rt_match = (sample_array[:, 1] >= peptide['Retention time'] - 3) & (
                            sample_array[:, 1] <= peptide['Retention time'] + 3)
                if True not in rt_match:
                    continue
                match = mz_match & rt_match & intensity_match

            if np.any(match):
                matched_indices = np.where(match)[0]
                if matched_indices.shape[0] > 1:
                    mz_diff = []
                    rt_diff = []
                    for idx in matched_indices:
                        temp_pep = sample_array[idx]
                        mz_diff.append(abs(temp_pep[0] - peptide['m/z']))
                        rt_diff.append(abs(temp_pep[1] - peptide['Retention time']))

                    mz_diff = np.array(mz_diff)
                    rt_diff = np.array(rt_diff)

                    min_mz_diff = np.min(mz_diff)
                    max_mz_diff = np.max(mz_diff)
                    min_rt_diff = np.min(rt_diff)
                    max_rt_diff = np.max(rt_diff)
                    if max_mz_diff == min_mz_diff:
                        mz_diff_normalized = np.zeros_like(mz_diff)
                    else:
                        mz_diff_normalized = (mz_diff - min_mz_diff) / (max_mz_diff - min_mz_diff)

                    if max_rt_diff == min_rt_diff:
                        rt_diff_normalized = np.zeros_like(rt_diff)
                    else:
                        rt_diff_normalized = (rt_diff - min_rt_diff) / (max_rt_diff - min_rt_diff)

                    score = mz_diff_normalized + rt_diff_normalized

                    best_idx = matched_indices[np.argmin(score)]
                    matched_indices = [best_idx]

                for index in matched_indices:
                    peptide_search_df.iloc[index, j] = sample_array[index, 2]
        j += 1

    #MBR
    peptide_search_df = peptide_search_df[(peptide_search_df != 0).any(axis=1)]
    zero_positions = np.where(peptide_search_df == 0)
    row_indices = peptide_search_df.index[zero_positions[0]]
    col_indices = zero_positions[1]
    m = -1
    n = 0
    for row, col in zip(row_indices, col_indices):
        value = feature_df.iloc[row, 3 * col + 2]
        m += 1
        if value == 0:
            continue
        peptide_search_df.iloc[zero_positions[0][m], col] = value
        n += 1

    out_path = os.path.join("E:\Benchmark-MV", "Benchmark-MV_results_peptide_matrix", feature_source, align_method,
                            "peptide_matrix_" + align_method + '_after_MBR' + ".csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    peptide_search_df.to_csv(out_path, index=False)
    print(align_method + " finished")


def collect_intensity_FC_maxquant(peptides_path, feature_dict):
    columns = ['Proteins', 'Intensity 1_C011_20211231_7300ng_293T_27', 'Intensity 1_C011_20211231_9100ng_293T_9', 'Intensity 1_C011_20211231_9700ng_293T_3',
               'Intensity 2_C008_20211231_7300ng_293T_27', 'Intensity 2_C008_20211231_9100ng_293T_9', 'Intensity 2_C008_20211231_9700ng_293T_3',
               'Intensity 3_C009_20211231_7300ng_293T_27', 'Intensity 3_C009_20211231_9100ng_293T_9', 'Intensity 3_C009_20211231_9700ng_293T_3']
    df = pd.read_csv(peptides_path, sep='\t', usecols=columns)
    df = df[df['Proteins'].str.contains('ECOLI', na=False)]
    df = df.drop('Proteins', axis=1)
    feature_df = feature_dict[0]
    area_df = feature_df.filter(like='area', axis=1)
    for n in range(0, len(df.columns)):
        peptide_area = df.iloc[:, n]
        feature_area = area_df.iloc[:, n]
        for index in peptide_area.index:
            area = peptide_area[index]
            if area == 0:
                continue
            intensity_match = np.abs(feature_area - area) / area < 1.0e-6
            if intensity_match.any():
                true_count = np.sum(intensity_match)
                print(f"intensity_match 中有 {true_count} 个 True 值")
                continue
            else:
                df.loc[index, df.columns[n]] = 0
    df = df.loc[~(df == 0).all(axis=1)]
    df.to_csv(r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\maxquant\peptides_MBR.csv", index=False)
    for column in df.columns:
        non_zero_count = (df[column] != 0).sum()
        print(non_zero_count)

def collect_maxquant_peptides(evidence_path):
    peptide_df = pd.read_csv(evidence_path, sep='\t', usecols=['Sequence', 'Proteins', 'Raw file', 'm/z', 'Retention time', 'Intensity', 'Score'])
    peptide_df = peptide_df[pd.notna(peptide_df['Intensity'])]
    peptide_df = peptide_df[peptide_df['Proteins'].str.contains('ECOLI', na=False)]
    grouped_peptide_df = peptide_df.groupby('Raw file')
    for raw_file, group in grouped_peptide_df:
        outpath = os.path.join(r"E:\Benchmark-MV\matched_peptide\maxquant_peptide", raw_file + '.csv')
        group.to_csv(outpath, sep=',', header=True, index=False)



def find_density_thresholds(data, threshold=0.1):
    kde = gaussian_kde(data.dropna())
    x = np.linspace(data.min(), data.max(), 1000)
    y = kde(x)

    peak_idx = y.argmax()
    left_idx = peak_idx
    while left_idx > 0 and y[left_idx] > threshold:
        left_idx -= 1
    left_x = x[left_idx] if left_idx > 0 else x[0]

    right_idx = peak_idx
    while right_idx < len(y) - 1 and y[right_idx] > threshold:
        right_idx += 1
    right_x = x[right_idx] if right_idx < len(y) - 1 else x[-1]

    return left_x, right_x

def get_threshold():
    peptide_matrix = pd.read_csv(r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\trend-aligner\peptide_matrix_trend-aligner_after_MBR.csv")
    column_base = peptide_matrix.columns[-3]
    threshold_dict = {}
    int_ratio = pd.DataFrame()
    for column_search in peptide_matrix.columns:
        if column_search == column_base:
                continue
        col = column_search + '/' + column_base
        ratio = peptide_matrix[column_search] / peptide_matrix[column_base]
        int_ratio[col] = np.log(ratio) / np.log(3)
        int_ratio = int_ratio.replace([0, np.inf, -np.inf], np.nan)
    n = 0
    for col in int_ratio.columns:
        kde = gaussian_kde(int_ratio[col].dropna())
        x = np.linspace(int_ratio[col].dropna().min(), int_ratio[col].dropna().max(), 500)
        peak = x[kde(x).argmax()]
        threshold_dict[n] = {
            'top_edge': peak + 0.3,
            'down_edge': peak - 0.3
        }
        n += 1
    return threshold_dict

def remove_error_and_collect_intensity(paths, methods, threshold_dict):
    for path, method in zip(paths, methods):
        peptide_matrix = pd.read_csv(path)
        peptide_matrix = peptide_matrix[peptide_matrix.iloc[:, -3] != 0]
        non_zero_counts = peptide_matrix.ne(0).sum()
        print(method)
        for count in non_zero_counts:
            print(count)
        column_base = peptide_matrix.columns[-3]
        int_ratio = pd.DataFrame()
        for column_search in peptide_matrix.columns:
            if column_search == column_base:
                continue
            col = column_search + '/' + column_base
            ratio = peptide_matrix[column_search] / peptide_matrix[column_base]
            int_ratio[col] = np.log(ratio) / np.log(3)
        int_ratio = int_ratio.replace([0, np.inf, -np.inf], np.nan)

        plt.figure(figsize=(8, 5), dpi=600)
        sns.boxplot(data=int_ratio)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(-5, 5)
        plt.title(f"{method}: Intensity ratio based on 2700R3")
        plt.tight_layout()
        output_path = f"E:\\workspace_plot\\peptide_intensity\\{method}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.show()

        #error_peptide_removal
        for index, row in int_ratio.iterrows():
            i = 0
            for col in row.index:
                value = row[col]
                if np.isnan(value):
                    i += 1
                    continue
                top_edge = threshold_dict[i]['top_edge']
                down_edge = threshold_dict[i]['down_edge']
                i += 1
                if down_edge <= value <= top_edge:
                    continue
                else:
                    int_ratio.loc[index, col] = np.nan

        plt.figure(figsize=(8, 5), dpi=600)
        sns.boxplot(data=int_ratio)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(-5, 5)
        plt.title(f"{method}: Intensity ratio based on 2700R3")
        plt.tight_layout()
        output_path = f"E:\\workspace_plot\\peptide_intensity\\{method}_removed.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.show()

        counts = int_ratio.notna().sum()
        n = 0
        for count in counts:
            if n == 6:
                print(peptide_matrix[column_base].ne(0).sum())
            print(count)
            n += 1



#workflow
#collect_maxquant_peptide_list
collect_maxquant_peptides(r"C:\Users\Nico\Downloads\combined\txt\evidence.txt")

peptide_folder = r"E:\Benchmark-MV\matched_peptide\maxquant_peptide"
peptide_paths = glob.glob(os.path.join(peptide_folder, "*.csv"))

feature_dict = filtering_feature([r"E:\Benchmark-MV\results\Benchmark-MV_aligned_trend-aligner.csv",
                                    r"E:\Benchmark-MV\deeprtalign_results\Benchmark-MV_aligned_deeprtalign.csv",
                                    r"E:\Benchmark-MV\openms_results\Benchmark-MV_aligned_openms.csv",
                                    r"E:\Benchmark-MV\xcms_results\Benchmark-MV_aligned_xcms_group.csv",
                                    r"E:\Benchmark-MV\xcms_results\Benchmark-MV_aligned_xcms_obiwarp.csv",
                                    r"E:\Benchmark-MV\mzmine2_results\Benchmark-MV_aligned_mzmine2_ransac.csv",
                                    r"E:\Benchmark-MV\mzmine2_results\Benchmark-MV_aligned_mzmine2_join.csv"
                                                ])

methods = ['trend-aligner', 'deeprtalign', 'openms', 'xcms_group', 'xcms_obiwarp', 'mzmine2_ransac', 'mzmine2_join']
n = 0
for feature_df in feature_dict.values():
    convert_feature_to_peptide('maxquant', methods[n], feature_df, peptide_paths, 'maxquant')
    n += 1

collect_intensity_FC_maxquant(r"E:\Benchmark-MV\maxquant_results\peptides.txt", feature_dict)


paths = [
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\trend-aligner\peptide_matrix_trend-aligner_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\deeprtalign\peptide_matrix_deeprtalign_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\openms\peptide_matrix_openms_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\xcms_group\peptide_matrix_xcms_group_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\xcms_obiwarp\peptide_matrix_xcms_obiwarp_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\mzmine2_ransac\peptide_matrix_mzmine2_ransac_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\mzmine2_join\peptide_matrix_mzmine2_join_after_MBR.csv",
        r"E:\Benchmark-MV\Benchmark-MV_results_peptide_matrix\maxquant\maxquant\peptides_MBR.csv"
        ]
methods = ['Trend-Aligner', 'DeepRTAlign', 'OpenMS', 'XCMS(Group)', 'XCMS(OBI-warp)', 'MZmine2(RANSAC)', 'MZmine2(Join)', 'MaxQuant']

threshold_dict = get_threshold()
remove_error_and_collect_intensity(paths, methods, threshold_dict)
