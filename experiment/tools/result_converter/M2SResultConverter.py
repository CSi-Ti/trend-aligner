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
import pandas as pd


class M2SResultConverter:
    """
    M2SResultConverter: Converts M2S alignment results into a format suitable for the evaluation code eval.py
                    (m/z_median, rt_median, intensity_median, #, sample1_m/z, sample1_rt, sample1_intensity ...)

    :param path: The path to the M2S alignment results M2S_datasetsMatched.xlsx file
    :param dataset_name: The name of the dataset, used to generate the output file name
    :param sample_names: A list of sample names; the order of all datasets and ground truth samples in the evaluation follows ASCII order
    :param save_folder: The folder for saving the output file; if empty, the output file will be saved in the same folder as the alignment results, default is empty
    """

    def __init__(self, path, dataset_name, sample_names, save_folder=None):
        self.path = path
        self.dataset_name = dataset_name
        self.save_folder = save_folder
        self.sample_names = sample_names

    def convert(self):
        try:
            xlsx_file = pd.ExcelFile(self.path)
            pairs_counts = int(len(xlsx_file.sheet_names) / 2)
            ref_1 = xlsx_file.parse('Ref_1')
            Target_1 = xlsx_file.parse('Target_1')

            result = ref_1[['mzmed', 'rtmed', 'fimed']]
            result = result.rename(columns={'mzmed': 'mzmed_ref_1', 'rtmed': 'rtmed_ref_1', 'fimed': 'fimed_ref_1'})

            for i in range(pairs_counts):
                ref_sample = xlsx_file.parse('Ref_' + str(i + 1))
                Target_sample = xlsx_file.parse('Target_' + str(i + 1))
                ref_sample = pd.DataFrame(ref_sample)
                Target_sample = pd.DataFrame(Target_sample)

                if len(Target_sample) == 0:
                    result['mzmed' + '_Target_' + str(i + 1)] = 0
                    result['rtmed' + '_Target_' + str(i + 1)] = 0
                    result['fimed' + '_Target_' + str(i + 1)] = 0
                    continue

                for j, row in ref_sample.iterrows():
                    result = pd.DataFrame(result)
                    indice = result[(result['mzmed_ref_1'] == row['mzmed']) &
                                    (result['rtmed_ref_1'] == row['rtmed']) &
                                    (result['fimed_ref_1'] == row['fimed'])].index

                    if not indice.empty:
                        print(i, j)
                        indice = indice.astype(int).values[0]
                        result.at[indice, 'mzmed' + '_Target_' + str(i + 1)] = Target_sample.at[j, 'mzmed']
                        result.at[indice, 'rtmed' + '_Target_' + str(i + 1)] = Target_sample.at[j, 'rtmed']
                        result.at[indice, 'fimed' + '_Target_' + str(i + 1)] = Target_sample.at[j, 'fimed']
                    else:
                        new_row = pd.Series({
                            'mzmed_ref_1': row['mzmed'],
                            'rtmed_ref_1': row['rtmed'],
                            'fimed_ref_1': row['fimed'],
                            'mzmed' + '_Target_' + str(i + 1): Target_sample.at[j, 'mzmed'],
                            'rtmed' + '_Target_' + str(i + 1): Target_sample.at[j, 'rtmed'],
                            'fimed' + '_Target_' + str(i + 1): Target_sample.at[j, 'fimed']
                        })
                        new_row = new_row.to_frame().T
                        result = result.append(new_row, ignore_index=True)

            colnames = [f"{sample}_{metric}" for sample in self.sample_names for metric in ['mz', 'rt', 'area']]
            result.columns = colnames

            result['mz_med'] = result.iloc[:, [0 + 3 * m for m in range(0, pairs_counts + 1)]].median(axis=1)
            result['rt_med'] = result.iloc[:, [1 + 3 * m for m in range(0, pairs_counts + 1)]].median(axis=1)
            result['area_med'] = result.iloc[:, [2 + 3 * m for m in range(0, pairs_counts + 1)]].median(axis=1)
            result['#'] = 0
            result.fillna(0, inplace=True)

            new_column_order = ['mz_med', 'rt_med', 'area_med', '#']
            result = result.reindex(columns=new_column_order + colnames)
            result.fillna(0, inplace=True)
            if self.save_folder:
                out_path = os.path.join(self.save_folder, f"{self.dataset_name}_aligned_M2S.csv")
            else:
                out_path = os.path.join(os.path.dirname(self.path), f"{self.dataset_name}_aligned_M2S.csv")
            result.to_csv(out_path, sep=',', index=False, header=True)
            print(f"{self.path} convert finished")
        except FileNotFoundError:
            print(f"Error: The file {self.path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
