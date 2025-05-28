# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren, Etienne Caron and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import pandas as pd
import numpy as np

class AntDASResultConverter:
    """
    AntDASResultConverter: Converts AntDAS alignment results into a format suitable for the evaluation code eval.py
                        (m/z_median, rt_median, intensity_median, #, sample1_m/z, sample1_rt, sample1_intensity ...)

    :param path: The path to the AntDAS alignment results BatchAnalysisResults4MetaboAnalyst file
    :param dataset_name: The name of the dataset, used to generate the output file name
    :param align_method: The alignment method in AntDAS, used to generate the output file name #group-based coarse+precise direct coarse+nearest
    :param sample_paths: All AntDAS original sample paths (.mzML_msePeakDectection.xlsx files), used to compare with the feature list in the
                        alignment results to remove features that were force-integrated during gap filling
    :param save_folder: The folder for saving the output file; if empty, the output file will be saved in the same folder as the alignment results, default is empty
    """

    def __init__(self, path, dataset_name, align_method, sample_paths, save_folder=None):
        self.path = path
        self.dataset_name = dataset_name
        self.align_method = align_method
        self.sample_paths = sample_paths
        self.save_folder = save_folder
        self.sample_files = {}
        self.sample_names = []

    def load_sample_files(self):
        for sample_path in self.sample_paths:
            name = os.path.basename(sample_path).split("_msePeakDectection.xlsx")[0]
            self.sample_names.append(name)
            self.sample_files[name] = pd.read_excel(sample_path, dtype=str)

    def convert(self):
        try:
            self.load_sample_files()
            result = []

            aligned_result = pd.read_csv(self.path, sep=',')
            for index, row in aligned_result.iterrows():
                if index == 0:
                    continue
                row_to_append = []
                n = 1
                for value in row.iloc[1:]:
                    name = aligned_result.columns[n]
                    n += 1
                    sample = self.sample_files[name]
                    matched_rows = sample.loc[sample['Area'] == value]
                    if matched_rows.empty:
                        print(n, "No matched") #Remove features that were force-integrated during gap filling in the original algorithm.
                        row_to_append.extend([0, 0, 0])
                        continue
                    row_to_append.append(matched_rows["m/z"].tolist()[0])
                    row_to_append.append(matched_rows["RT"].tolist()[0])
                    row_to_append.append(matched_rows["Area"].tolist()[0])

                result.append(row_to_append)

            result_df = pd.DataFrame(result)
            result_df = result_df.apply(pd.to_numeric, errors='coerce')
            columns = []
            for name in aligned_result.columns[1:].tolist():
                columns.append(name + "_mz")
                columns.append(name + "_rt")
                columns.append(name + "_area")
            result_df.columns = columns
            result_df.replace(0, np.nan, inplace=True)
            result_df['mz_med'] = result_df.iloc[:, [0 + 3 * m for m in range(0, len(self.sample_paths))]].median(axis=1)
            result_df['rt_med'] = result_df.iloc[:, [1 + 3 * m for m in range(0, len(self.sample_paths))]].median(axis=1)
            result_df['area_med'] = result_df.iloc[:, [2 + 3 * m for m in range(0, len(self.sample_paths))]].median(axis=1)
            result_df['#'] = 0
            result_df.fillna(0, inplace=True)

            column_names = ['mz_med', 'rt_med', 'area_med', '#']
            for name in self.sample_names:
                column_names.append(name + "_mz")
                column_names.append(name + "_rt")
                column_names.append(name + "_area")
            result_df = result_df.reindex(columns=column_names)
            if self.save_folder:
                out_path = os.path.join(self.save_folder, f"{self.dataset_name}_aligned_AntDAS_{self.align_method}.csv")
            else:
                out_path = os.path.join(os.path.dirname(self.path), f"{self.dataset_name}_aligned_AntDAS_{self.align_method}.csv")
            result_df.to_csv(out_path, sep=',', index=False, header=True)
            print(f"{self.path} convert finished.")
        except FileNotFoundError:
            print(f"Error: The file {self.path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")