# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren, Etienne Caron and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import glob
import pandas as pd
import math
import os
import numpy as np


def convert_feature_to_peptide(feature_source, align_method, aligned_feature_path,  peptide_paths, separator=','):
    feature_df = pd.read_csv(aligned_feature_path, sep=separator)
    peptide_dict = {}
    for index, path in enumerate(peptide_paths):
        peptide_df = pd.read_csv(path, sep=",",)
        peptide_dict[index] = peptide_df
    feature_df = feature_df.iloc[:, 4:].replace(-1, 0)

    step = 3
    num_columns = int(len(feature_df.columns) / 3)
    num_rows = int(len(feature_df.index))
    col_names = ["10R1", "10R2", "10R3", "15R1", "15R2", "15R3", "20R1", "20R2", "20R3", "25R1", "25R2", "25R3"]
    peptide_search_df = pd.DataFrame(0, index=range(num_rows), columns=range(num_columns))
    peptide_search_df.columns = col_names

    for i in range(0, 36, step):
        sample_df = feature_df.iloc[:, i:i + step]
        sample_array = sample_df.to_numpy()
        for n, peptide_df in peptide_dict.items():
            library_array = peptide_df[['m/z', 'Retention time', 'Intensity']].to_numpy()

            for peptide in library_array:
                intensity_match = np.abs(sample_array[:, 2] - peptide[2]) / peptide[2] < 1.0e-6
                if True not in intensity_match:
                    continue
                mz_match = (sample_array[:, 0] >= peptide[0] - 0.1) & (sample_array[:, 0] <= peptide[0] + 0.1)
                if True not in mz_match:
                    continue
                rt_match = (sample_array[:, 1] >= peptide[1] - 3) & (sample_array[:, 1] <= peptide[1] + 3)
                if True not in rt_match:
                    continue

                match = mz_match & rt_match & intensity_match
                if np.any(match):
                    if len(np.where(match)) > 1:
                        print(len(np.where(match)))
                    matched_indices = np.where(match)[0]
                    for index in matched_indices:
                        peptide_search_df.iloc[index, n] = sample_array[index, 2]


    out_path = os.path.join("E:\workspace", "Benchmark_FC_results_peptide_matrix", feature_source, align_method, "peptide_matrix_" + align_method + ".csv")
    peptide_search_df.to_csv(out_path, index=False)

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

    out_path = os.path.join("E:\workspace", "Benchmark_FC_results_peptide_matrix", feature_source, align_method,
                            "peptide_matrix_" + align_method + '_after_MBR' + ".csv")
    peptide_search_df.to_csv(out_path, index=False)
    collect_intensity_FC(peptide_search_df, align_method)
    print(align_method + " finished")
    return peptide_search_df


def collect_intensity_FC(aligned_peptide_matrix, align_method):
    sample_names = ["10R1", "10R2", "10R3", "15R1", "15R2", "15R3", "20R1", "20R2", "20R3", "25R1", "25R2", "25R3"]
    aligned_peptide_matrix = pd.DataFrame(aligned_peptide_matrix)
    for m in range(3, 12):
        n = m - 3
        if m > 5:
            n = m - 6
        if m > 8:
            n = m -9
        sample_1 = sample_names[m]
        sample_2 = sample_names[n]
        collect_matrix = aligned_peptide_matrix.iloc[:, [m, n]]
        collect_matrix = collect_matrix[(collect_matrix != 0).all(axis=1)]
        result = {'sample_1': [sample_1] * len(collect_matrix),
                  'sample_1_intensity': collect_matrix.iloc[:, 0],
                  'sample_2': [sample_2] * len(collect_matrix),
                  'sample_2_intensity': collect_matrix.iloc[:, 1]}
        collect_matrix = pd.DataFrame(result)
        collect_matrix['FC'] = collect_matrix.apply(
            lambda row: math.log2(row['sample_1_intensity']) - math.log2(row['sample_2_intensity']), axis=1)
        out_path = os.path.join("E:\workspace", "Benchmark_FC_results_peptide_matrix", 'maxquant', align_method, 'intensity_FC_' + align_method + '_' + sample_1 + '_' + sample_2 + '_' + '.csv')
        collect_matrix.to_csv(out_path, index=False, header=True)



def collect_intensity_FC_maxquant(peptides_path, MBR=False):
    columns = ['Proteins', 'Intensity 10_R1', 'Intensity 10_R2', 'Intensity 10_R3', 'Intensity 15_R1', 'Intensity 15_R2', 'Intensity 15_R3', 'Intensity 20_R1', 'Intensity 20_R2', 'Intensity 20_R3', 'Intensity 25_R1', 'Intensity 25_R2', 'Intensity 25_R3']
    df = pd.read_csv(peptides_path, sep='\t', usecols=columns)
    df = df[df['Proteins'].str.contains('ECOLI', na=False)]
    df = df.drop('Proteins', axis=1)
    df[df < 1.0e8] = 0
    df = df.loc[~(df == 0).all(axis=1)]
    if MBR:
        df.to_csv(os.path.join(os.path.dirname(peptides_path), 'peptides_MBR.csv'), index=False)
    else:
        df.to_csv(os.path.join(os.path.dirname(peptides_path), 'peptides.csv'), index=False)
    for column in df.columns:
        non_zero_count = (df[column] != 0).sum()
        print(non_zero_count)

    if MBR:
        df.columns = ["10R1", "10R2", "10R3", "15R1", "15R2", "15R3", "20R1", "20R2", "20R3", "25R1", "25R2", "25R3"]
        collect_intensity_FC(df, 'maxquant_workflow')






def collect_maxquant_peptides(evidence_path):
    peptide_df = pd.read_csv(evidence_path, sep='\t', usecols=['Sequence', 'Proteins', 'Raw file', 'm/z', 'Retention time', 'Intensity', 'Score'])
    peptide_df = peptide_df[pd.notna(peptide_df['Intensity'])]
    peptide_df = peptide_df[peptide_df['Proteins'].str.contains('ECOLI', na=False)]
    grouped_peptide_df = peptide_df.groupby('Raw file')
    for raw_file, group in grouped_peptide_df:
        outpath = os.path.join('E:\workspace\Benchmark_FC\matched_peptide\maxquant_peptide\\test', raw_file + '.csv')
        group.to_csv(outpath, sep=',', header=True, index=False)



#workflow
peptide_folder = r"E:\workspace\Benchmark_FC\matched_peptide\maxquant_peptide"

#collect_maxquant_peptide_list
# collect_maxquant_peptides(r"E:\workspace\Benchmark_FC\maxquant_output(E.coil)\combined\txt\evidence.txt")

#
peptide_paths = glob.glob(os.path.join(peptide_folder, "*.csv"))
# convert_feature_to_peptide("maxquant", "mzmine2_join", r"E:\workspace\Benchmark_FC_results_maxquant\mzmine2\Benchmark_FC_aligned_mzmine2_join.csv", peptide_paths)
# convert_feature_to_peptide("maxquant", "mzmine2_ransac", r"E:\workspace\Benchmark_FC_results_maxquant\mzmine2\Benchmark_FC_aligned_mzmine2_ransac.csv", peptide_paths)
# convert_feature_to_peptide("maxquant", "openms", r"E:\workspace\Benchmark_FC_results_maxquant\openms\Benchmark_FC_aligned_openms.csv", peptide_paths)
# convert_feature_to_peptide("maxquant", "deeprtalign", r"E:\workspace\Benchmark_FC_results_maxquant\deeprtalign\Benchmark_FC_aligned_deeprtalign.csv", peptide_paths)
convert_feature_to_peptide("maxquant", "trend-aligner", r"E:\workspace\Benchmark_FC_results_maxquant\trend-aligner\Benchmark_FC_aligned_trend-aligner.csv", peptide_paths)
# convert_feature_to_peptide("maxquant", "xcms_group", r"E:\workspace\Benchmark_FC_results_maxquant\xcms\Benchmark_FC_aligned_xcms_group.csv", peptide_paths)
# convert_feature_to_peptide("maxquant", "xcms_obiwarp", r"E:\workspace\Benchmark_FC_results_maxquant\xcms\Benchmark_FC_aligned_xcms_obiwarp.csv", peptide_paths)


# collect_intensity_FC_maxquant(r"E:\workspace\Benchmark_FC_results_peptide_matrix\maxquant\maxquant_workflow\peptides_MBR.txt", True)