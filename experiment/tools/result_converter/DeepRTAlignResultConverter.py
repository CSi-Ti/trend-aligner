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
import numpy as np


class DeepRTAlignResultConverter:
    """
    DeepRTAlignResultConverter: Converts DeepRTAlign alignment results into a format suitable for the evaluation code eval.py
                            (m/z_median, rt_median, intensity_median, #, sample1_m/z, sample1_rt, sample1_intensity ...)

    :param path: The path to the DeepRTAlign alignment results information_target.csv file
    :param dataset_name: The name of the dataset, used to generate the output file name
    :param sample_names: A list of sample names; the order of all datasets and ground truth samples in the evaluation follows ASCII order
    :param save_folder: The folder for saving the output file; if empty, the output file will be saved in the same folder as the alignment results, default is empty
    :param skip_line: The number of lines to skip, default is 1
    :param separator: The separator for the file, default is ','
    """

    def __init__(self, path, dataset_name, sample_names, save_folder=None, skip_line=1, separator=','):
        self.path = path
        self.dataset_name = dataset_name
        self.save_folder = save_folder
        self.sample_names = sample_names
        self.skip_line = skip_line
        self.separator = separator

    def convert(self):
        try:
            with open(self.path, 'r') as result_file:
                for i in range(self.skip_line):
                    header = result_file.readline().strip().split(self.separator)
                aligned_matrix_deeprtalign = np.array([line.strip().split(self.separator) for line in result_file])

            mz_index = header.index("mz")
            rt_index = header.index("time")
            intensity_index = header.index("intensity")
            sample_index = header.index("sample")
            group_index = header.index("group\n")

            mzs_data = aligned_matrix_deeprtalign[:, mz_index].astype(np.float32)
            rts_data = aligned_matrix_deeprtalign[:, rt_index].astype(np.float32)
            intensity_data = aligned_matrix_deeprtalign[:, intensity_index].astype(np.float32)
            sample_data = aligned_matrix_deeprtalign[:, sample_index]
            group_data = aligned_matrix_deeprtalign[:, group_index]

            group_dict = {}
            for group, intensity, sample, mz, rt in zip(group_data, intensity_data, sample_data, mzs_data, rts_data):
                if group not in group_dict:
                    group_dict[group] = {'mzs': [], 'rts': [], 'intensities': [], 'samples': []}
                group_dict[group]['mzs'].append(mz)
                group_dict[group]['rts'].append(rt)
                group_dict[group]['intensities'].append(intensity)
                group_dict[group]['samples'].append(sample)

            all_rows = []
            for group, data in group_dict.items():
                row = []
                samples = data['samples']
                mzs = data['mzs']
                rts = data['rts']
                intensities = data['intensities']
                mzs_data_median = np.median(data['mzs'])
                rts_data_median = np.median(data['rts'])
                intensities_data_median = np.median(data['intensities'])
                row.append(mzs_data_median)
                row.append(rts_data_median)
                row.append(intensities_data_median)
                row.append(0)
                for sample in self.sample_names:
                    if sample in samples:
                        index = samples.index(sample)
                        row.extend([mzs[index], rts[index], intensities[index]])
                    else:
                        row.extend([0, 0, 0])
                all_rows.append(row)

            result_array = np.array(all_rows)
            result_array = result_array[np.argsort(result_array[:, 0])]
            if self.save_folder:
                out_path = os.path.join(self.save_folder, f"{self.dataset_name}_aligned_deeprtalign.csv")
            else:
                out_path = os.path.join(os.path.dirname(self.path), f"{self.dataset_name}_aligned_deeprtalign.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savetxt(out_path, result_array, delimiter=',', fmt='%s')
            print(f"{self.path} convert finished")
        except FileNotFoundError:
            print(f"Error: The file {self.path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")