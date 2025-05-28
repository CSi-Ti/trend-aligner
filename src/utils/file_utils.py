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
import csv
import glob
import numpy as np


def load_feature_list(file_path, skip_line, separator, mz_col_idx, rt_col_idx, area_col_idx):
    if separator is None:
        if file_path.endswith('.tsv') or file_path.endswith('.txt') or file_path.endswith('.hills.csv'):
            separator = '\t'
        else:
            separator = ','

    feature_list_file = open(file_path, 'r')
    for j in range(skip_line):
        header = feature_list_file.readline()
    file_data = np.array([line.strip().replace('"', '').split(separator) for line in feature_list_file])
    feature_list = file_data[:, (mz_col_idx, rt_col_idx, area_col_idx)].astype(np.float32)
    return feature_list


def load_feature_lists(file_folder, skip_line, separator, mz_col_idx, rt_col_idx, area_col_idx):
    file_paths = glob.glob(os.path.join(file_folder, '*'))

    feature_lists = [load_feature_list(path, skip_line, separator, mz_col_idx, rt_col_idx, area_col_idx)
                     for path in file_paths]
    file_names = [os.path.basename(path) for path in file_paths]

    return np.array(feature_lists, dtype=object), np.array(file_names, dtype=object)


def save_params(save_path, result_file_params, coarse_alignment_params, fine_matching_params):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file = open(os.path.join(save_path, 'params.txt'), 'w')

    # ResultFileReadingParams
    file.write('# Result File Reading Params' + '\n')
    file.write('\tresult_file_path: ' + result_file_params.feature_list_folder_path + '\n')
    file.write('\tskip_line: ' + str(result_file_params.skip_line) + '\n')
    file.write('\tmz_col_num: ' + str(result_file_params.mz_col_idx + 1) + '\n')
    file.write('\trt_col_num: ' + str(result_file_params.rt_col_idx + 1) + '\n')
    file.write('\tarea_col_num: ' + str(result_file_params.area_col_idx + 1) + '\n')
    file.write(os.linesep)

    # CoarseRegistrationParams
    file.write('# Coarse Registration Params' + '\n')
    file.write('\tcentric_idx: ' + str(coarse_alignment_params.centric_idx) + '\n')
    file.write('\tmz_tolerance: ' + str(coarse_alignment_params.mz_tolerance) + '\n')
    file.write('\trt_tolerance: ' + str(coarse_alignment_params.rt_tolerance) + '\n')
    file.write('\tuse_ppm: ' + str(coarse_alignment_params.use_ppm) + '\n')
    file.write('\tfrom_rt: ' + str(coarse_alignment_params.from_rt) + '\n')
    file.write('\tto_rt: ' + str(coarse_alignment_params.to_rt) + '\n')
    file.write('\tfrac: ' + str(coarse_alignment_params.frac) + '\n')
    file.write(os.linesep)

    # FineAssignmentParams
    file.write('# Fine Assignment Params' + '\n')
    file.write('\tbeam_mz_tol: ' + str(fine_matching_params.beam_mz_tol) + '\n')
    file.write('\tbeam_rt_tol: ' + str(fine_matching_params.beam_rt_tol) + '\n')
    file.write('\tmatch_mz_tol: ' + str(fine_matching_params.match_mz_tol) + '\n')
    file.write('\tmatch_rt_tol: ' + str(fine_matching_params.match_rt_tol) + '\n')
    file.write('\tmax_rt_tol: ' + str(fine_matching_params.max_rt_tol) + '\n')
    file.write('\tuse_ppm: ' + str(fine_matching_params.use_ppm) + '\n')
    file.write('\tbeam_width: ' + str(fine_matching_params.beam_width) + '\n')
    file.write('\tmz_factor: ' + str(fine_matching_params.mz_factor) + '\n')
    file.write('\trt_factor: ' + str(fine_matching_params.rt_factor) + '\n')
    file.write('\tarea_factor: ' + str(fine_matching_params.area_factor) + '\n')
    file.write(os.linesep)

    file.close()


def save_alignment_results(aligned_feature_matrix, file_names, save_path, save_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = os.path.join(save_path, save_name)
    file = open(file_path, 'w')
    writer = csv.writer(file, dialect='unix', quoting=csv.QUOTE_NONE, quotechar='')
    first_row = ['mz', 'rt', 'area', '#']
    for file_name in file_names:
        first_row += [file_name + "_mz", file_name + "_rt", file_name + "_area"]
    writer.writerow(first_row)

    rt_indices = [3 * i + 1 for i in range(len(file_names))]
    for aligned_feature_row in aligned_feature_matrix:
        rt_row = aligned_feature_row[rt_indices]
        median_value = rt_row[rt_row != -1][np.argsort(rt_row[rt_row != -1])[len(rt_row[rt_row != -1]) // 2]]
        median_rt_idx = rt_indices[np.where(rt_row == median_value)[0][0]]
        median_feature = aligned_feature_row[[median_rt_idx - 1, median_rt_idx, median_rt_idx + 1]]

        row = np.hstack([median_feature, np.array([0]), aligned_feature_row])
        writer.writerow(row)
    file.close()
    return file_path