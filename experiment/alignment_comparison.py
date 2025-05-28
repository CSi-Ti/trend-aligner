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
import os.path
import warnings
import numpy as np
import pandas as pd


def load_lib(lib_path, sample_names):
    warnings.simplefilter('ignore')
    data = pd.read_csv(lib_path)
    ana_idx = data['Type'] == 'ANALYTES'
    lib_matrix = []
    for i, sample in enumerate(sample_names):
        mzs = np.array(data[sample + '_Mz'][ana_idx])
        rts = np.array(data[sample + '_Rt'][ana_idx])
        lib_matrix.append(np.vstack([mzs, rts]).transpose())
    lib_matrix = np.array(lib_matrix)
    sorted_idx = np.argsort(np.mean(lib_matrix[:, :, 0], axis=0))
    lib_matrix = lib_matrix[:, sorted_idx, :]
    return lib_matrix

def load_features(result_paths, separator, skip_line, mz_col_idx, rt_col_idx, area_col_idx):
    result_matrix = []
    for idx, result_path in enumerate(result_paths):
        result_file = open(result_path, 'r')
        for i in range(skip_line):
            header = result_file.readline().split(separator)
        result_data = np.array([line.strip().split(separator) for line in result_file])
        mzs = result_data[:, mz_col_idx].astype(np.float32)
        rts = result_data[:, rt_col_idx].astype(np.float32)
        areas = result_data[:, area_col_idx].astype(np.float32)
        sorted_idx = np.argsort(mzs)
        result = np.vstack([mzs[sorted_idx], rts[sorted_idx], areas[sorted_idx], np.ones(len(mzs)) * idx]).transpose()
        available_idx = [True] * len(result)
        for i in range(1, len(result)):
            if result[i][0] == result[i - 1][0] and result[i][1] == result[i - 1][1]:
                # find max
                available_idx[i] = False
        result_matrix.append(result[available_idx])

    return result_matrix

def load_aligned_result(result_path, separator=',', skip_line=1):
    result_file = open(result_path, 'r')
    for i in range(skip_line):
        header = result_file.readline().split(separator)
    aligned_matrix = np.array([line.strip().split(separator) for line in result_file]).astype(np.float32)
    aligned_matrix = aligned_matrix[np.argsort(aligned_matrix[:, 0])]
    return aligned_matrix

def match_feature_to_lib(lib_matrix, result_matrix, mz_tolerance, use_ppm, rt_tolerance):
    match_matrix = np.ones(lib_matrix.shape[:2]).astype(np.int32) * -1
    mz_error_matrix = np.zeros(lib_matrix.shape[:2])
    rt_error_matrix = np.zeros(lib_matrix.shape[:2])
    for i in range(len(lib_matrix)):
        lib_data = lib_matrix[i]
        sorted_lib_idx = np.argsort(lib_data[:, 0])
        result_data = result_matrix[i]
        result_idx = 0
        for j in range(len(lib_data)):
            lib_idx = sorted_lib_idx[j]
            tmp_mz_tolerance = mz_tolerance
            if use_ppm:
                tmp_mz_tolerance = lib_data[lib_idx][0] * mz_tolerance * 1e-6
            mz_start = lib_data[lib_idx][0] - tmp_mz_tolerance
            mz_end = lib_data[lib_idx][0] + tmp_mz_tolerance
            if result_data[result_idx][0] > mz_end:
                continue
            while result_idx < len(result_data) and result_data[result_idx][0] < mz_start:
                result_idx += 1
            if result_idx >= len(result_data):
                break

            rt_start = lib_data[lib_idx][1] - rt_tolerance
            rt_end = lib_data[lib_idx][1] + rt_tolerance
            match_idx = -1
            min_dist = float('inf')
            mz_error = 0
            rt_error = 0
            for result_iter_idx in range(result_idx, len(result_data)):
                if result_data[result_iter_idx][0] > mz_end:
                    break
                if (result_data[result_iter_idx][1] < rt_start) \
                        or (result_data[result_iter_idx][1] > rt_end):
                    continue
                tmp_rt_error = abs(result_data[result_iter_idx][1] - lib_data[lib_idx][1])
                tmp_mz_error = abs(result_data[result_iter_idx][0] - lib_data[lib_idx][0])
                dist = np.square(tmp_rt_error / rt_tolerance) + np.square(tmp_mz_error / mz_tolerance)
                if dist < min_dist:
                    min_dist = dist
                    mz_error = tmp_mz_error
                    rt_error = tmp_rt_error
                    match_idx = result_iter_idx
            match_matrix[i, lib_idx] = match_idx
            mz_error_matrix[i, lib_idx] = mz_error
            rt_error_matrix[i, lib_idx] = rt_error
    return match_matrix, mz_error_matrix, rt_error_matrix

def match_feature_to_aligned(aligned_matrix, result_matrix, mz_tolerance, use_ppm, rt_tolerance):
    match_matrix = np.ones((aligned_matrix.shape[0], int(aligned_matrix.shape[1] / 3))).astype(np.int32) * -1
    for i in range(len(result_matrix)):
        sample_result = result_matrix[i]
        aligned_result = aligned_matrix[:, [3 * i, 1 + 3 * i, 2 + 3 * i]]
        sorted_aligned_idx = np.argsort(aligned_result[:, 0])
        result_idx = 0
        for j in range(len(aligned_result)):
            aligned_idx = sorted_aligned_idx[j]
            target_area = aligned_result[aligned_idx][2]
            if target_area < 1e-6:
                continue
            tmp_mz_tolerance = mz_tolerance
            if use_ppm:
                tmp_mz_tolerance = aligned_result[aligned_idx][0] * mz_tolerance * 1e-6
            mz_start = aligned_result[aligned_idx][0] - tmp_mz_tolerance
            mz_end = aligned_result[aligned_idx][0] + tmp_mz_tolerance
            if sample_result[result_idx][0] > mz_end:
                continue
            while result_idx < len(sample_result) and sample_result[result_idx][0] < mz_start:
                result_idx += 1
            if result_idx >= len(sample_result):
                break

            rt_start = aligned_result[aligned_idx][1] - rt_tolerance
            rt_end = aligned_result[aligned_idx][1] + rt_tolerance
            for result_iter_idx in range(result_idx, len(sample_result)):
                if sample_result[result_iter_idx][0] > mz_end:
                    break
                if (sample_result[result_iter_idx][1] < rt_start) \
                        or (sample_result[result_iter_idx][1] > rt_end):
                    continue
                if abs(sample_result[result_iter_idx][2] - target_area) < 1e-6:
                    match_matrix[aligned_idx, i] = result_iter_idx
                    break
    return match_matrix

def eval_alignment_performance(lib_match_matrix, align_match_matrix):
    eval_result = []

    for i in range(len(lib_match_matrix[0])):
        # 970
        lib_match_assignment = lib_match_matrix[:, i]
        if sum(lib_match_assignment != -1) < len(lib_match_matrix) / 2:
            lib_match_assignment = np.ones(len(lib_match_matrix)) * -1
        matched_idxes = []
        for j in range(len(lib_match_matrix)):
            if lib_match_assignment[j] == -1:
                continue
            matched_idx = np.where(align_match_matrix[:, j] == lib_match_assignment[j])[0]
            if len(matched_idx) > 0:
                # matched_idxes.append(matched_idx[0])
                matched_idxes += matched_idx.tolist()
        if len(matched_idxes) > 0:
            max_idx = np.argmax(np.bincount(matched_idxes))
            max_assignment = align_match_matrix[max_idx]
        else:
            max_assignment = np.ones(len(lib_match_matrix)).astype(np.int32) * -1
        tp = np.sum((max_assignment == lib_match_assignment) * (max_assignment != -1))
        fp = np.sum(max_assignment[max_assignment != -1] != lib_match_assignment[max_assignment != -1])
        tn = np.sum((max_assignment == -1) * (lib_match_assignment == -1))
        fn = np.sum((max_assignment == -1) * (lib_match_assignment != -1))
        tt = np.sum((max_assignment != -1) * (lib_match_assignment == -1))
        eval_result.append([tp, fp, tn, fn, tt])
    eval_result = np.array(eval_result)

    total_accuracy = np.sum(eval_result[:, [0, 2]]) / np.sum(eval_result[:, 0:4])
    total_precision = np.sum(eval_result[:, 0]) / np.sum(eval_result[:, [0, 1]])
    total_recall = np.sum(eval_result[:, 0]) / np.sum(eval_result[:, [0, 3]])
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    comp_acc = np.sum(eval_result[:, 0] + eval_result[:, 2] == len(lib_match_matrix)) / len(eval_result)
    total_scores = [total_accuracy, total_precision, total_recall, total_f1, comp_acc]
    print(np.sum(eval_result, axis=0)[:4].tolist() + total_scores)
    return eval_result


def eval(lib_matrix, result_matrix, aligned_matrix, mz_tolerance=0.0001, rt_tolerance=0.0001):
    lib_match_matrix, mz_error_matrix, rt_error_matrix = match_feature_to_lib(lib_matrix, result_matrix, mz_tolerance=mz_tolerance, use_ppm=False, rt_tolerance=rt_tolerance)#The tolerance here is used to assess the differences between software-extracted feature and metapro-extracted feature
    align_match_matrix = match_feature_to_aligned(aligned_matrix, result_matrix, mz_tolerance=0.5, use_ppm=False, rt_tolerance=5)#The tolerance here is used to assess the feature differences in the sample feature list before and after alignment
    eval_alignment_performance(lib_match_matrix, align_match_matrix)



if __name__ == '__main__':

    result_root_path = 'E:/workspace'

    MTBLS733_QE_HF_lib_path = os.path.join(result_root_path, 'MTBLS733_QE_HF', 'MTBLS733_QE_HF_annotated.csv')
    MTBLS733_QE_HF_samples = ['SA1', 'SA2', 'SA3', 'SA4', 'SA5', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5']

    MTBLS736_TripleTOF_6600_lib_path = os.path.join(result_root_path, 'MTBLS736_TripleTOF_6600',
                                                    'MTBLS736_TripleTOF_6600_annotated.csv')
    MTBLS736_TripleTOF_6600_samples = ['SampleA_1', 'SampleA_2', 'SampleA_3', 'SampleA_4', 'SampleB_1', 'SampleB_2',
                                       'SampleB_3', 'SampleB_4']
    MTBLS3038_NEG_lib_path = os.path.join(result_root_path, 'MTBLS3038_NEG', 'MTBLS3038_NEG_annotated.csv')
    MTBLS3038_NEG_samples = ["12-1G", "12-1M", "12-2G", "12-2M", "12-3G", "12-3M", "12-4G", "12-4M",
                             "17-1G", "17-1M", "17-2G", "17-2M", "17-3G", "17-3M", "17-4G", "17-4M",
                             "2-1G", "2-1M", "2-2G", "2-2M", "2-3G", "2-3M", "2-4G", "2-4M",
                             "21-1G", "21-1M", "21-2G", "21-2M", "21-3G", "21-3M", "21-4G", "21-4M",
                             "7-1G", "7-1M", "7-2G", "7-2M", "7-3G", "7-3M", "7-4G", "7-4M",
                             "QC01", "QC02", "QC03", "QC04", "QC05"]

    MTBLS3038_POS_lib_path = os.path.join(result_root_path, 'MTBLS3038_POS', 'MTBLS3038_POS_annotated.csv')
    MTBLS3038_POS_samples = ["12-1G", "12-1M", "12-2G", "12-2M", "12-3G", "12-3M", "12-4G", "12-4M",
                             "17-1G", "17-1M", "17-2G", "17-2M", "17-3G", "17-3M", "17-4G", "17-4M",
                             "2-1G", "2-1M", "2-2G", "2-2M", "2-3G", "2-3M", "2-4G", "2-4M",
                             "21-1G", "21-1M", "21-2G", "21-2M", "21-3G", "21-3M", "21-4G", "21-4M",
                             "7-1G", "7-1M", "7-2G", "7-2M", "7-3G", "7-3M", "7-4G", "7-4M",
                             "QC01", "QC02", "QC03", "QC04", "QC05"]

    MTBLS5430_Lip_NEG_lib_path = os.path.join(result_root_path, 'MTBLS5430_Lip_NEG', 'MTBLS5430_Lip_NEG_annotated.csv')
    MTBLS5430_Lip_NEG_samples = ['BA_1h_1', 'BA_1h_2', 'BA_1h_3', 'BA_24h_1', 'BA_24h_2', 'BA_24h_3', 'BA_24h_4',
                                 'BA_24h_5', 'BA_3h_1', 'BA_3h_2', 'BA_3h_3', 'BA_6h_1', 'BA_6h_2',
                                 'BA_6h_3', 'BA_Cont1', 'BA_Cont2', 'BA_Cont3', 'DR_1h_1', 'DR_1h_2', 'DR_1h_3',
                                 'DR_24h_1', 'DR_24h_2', 'DR_24h_3', 'DR_3h_1', 'DR_3h_2', 'DR_3h_3',
                                 'DR_6h_1', 'DR_6h_2', 'DR_6h_3', 'DR_Cont1', 'DR_Cont2', 'DR_Cont3', 'QC1', 'QC2',
                                 'QC3', 'QC4', 'QC5', 'QC6']


    MTBLS5430_Lip_POS_lib_path = os.path.join(result_root_path, 'MTBLS5430_Lip_POS', 'MTBLS5430_Lip_POS_annotated.csv')
    MTBLS5430_Lip_POS_samples = ['BA_1h_1', 'BA_1h_2', 'BA_1h_3', 'BA_24h_1', 'BA_24h_2', 'BA_24h_3', 'BA_24h_4',
                                 'BA_24h_5', 'BA_3h_1', 'BA_3h_2', 'BA_3h_3', 'BA_6h_1', 'BA_6h_2',
                                 'BA_6h_3', 'BA_Cont1', 'BA_Cont2', 'BA_Cont3', 'DR_1h_1', 'DR_1h_2', 'DR_1h_3',
                                 'DR_24h_1', 'DR_24h_2', 'DR_24h_3', 'DR_3h_1', 'DR_3h_2', 'DR_3h_3',
                                 'DR_6h_1', 'DR_6h_2', 'DR_6h_3', 'DR_Cont1', 'DR_Cont2', 'DR_Cont3', 'QC1', 'QC2',
                                 'QC3', 'QC4', 'QC5', 'QC6']


    MTBLS5430_Metabo_NEG_lib_path = os.path.join(result_root_path, 'MTBLS5430_Metabo_NEG',
                                                 'MTBLS5430_Metabo_NEG_annotated.csv')
    MTBLS5430_Metabo_NEG_samples = ['BA_1h_1', 'BA_1h_2', 'BA_1h_3', 'BA_24h_1', 'BA_24h_2', 'BA_24h_3', 'BA_24h_4',
                                    'BA_24h_5', 'BA_3h_1', 'BA_3h_2', 'BA_3h_3', 'BA_6h_1', 'BA_6h_2',
                                    'BA_6h_3', 'BA_Cont1', 'BA_Cont2', 'BA_Cont3', 'DR_1h_1', 'DR_1h_2', 'DR_1h_3',
                                    'DR_24h_1', 'DR_24h_2', 'DR_24h_3', 'DR_3h_1', 'DR_3h_2', 'DR_3h_3',
                                    'DR_6h_1', 'DR_6h_2', 'DR_6h_3', 'DR_Cont1', 'DR_Cont2', 'DR_Cont3', 'QC1', 'QC2',
                                    'QC3', 'QC4', 'QC5', 'QC6']

    MTBLS5430_Metabo_POS_lib_path = os.path.join(result_root_path, 'MTBLS5430_Metabo_POS',
                                                 'MTBLS5430_Metabo_POS_annotated.csv')
    MTBLS5430_Metabo_POS_samples = ['BA_1h_1', 'BA_1h_2', 'BA_1h_3', 'BA_24h_1', 'BA_24h_2', 'BA_24h_3', 'BA_24h_4',
                                    'BA_24h_5', 'BA_3h_1', 'BA_3h_2', 'BA_3h_3', 'BA_6h_1', 'BA_6h_2',
                                    'BA_6h_3', 'BA_Cont1', 'BA_Cont2', 'BA_Cont3', 'DR_1h_1', 'DR_1h_2', 'DR_1h_3',
                                    'DR_24h_1', 'DR_24h_2', 'DR_24h_3', 'DR_3h_1', 'DR_3h_2', 'DR_3h_3',
                                    'DR_6h_1', 'DR_6h_2', 'DR_6h_3', 'DR_Cont1', 'DR_Cont2', 'DR_Cont3', 'QC1', 'QC2',
                                    'QC3', 'QC4', 'QC5', 'QC6']

    AT_lib_path = os.path.join(result_root_path, 'AT', 'AT_annotated.csv')
    AT_samples = ['0_1', '0_2', '0_3', '24_1', '24_2', '24_3', '36_1', '36_2', '36_3', '48_1', '48_2', '48_3', '60_1',
                  '60_2', '60_3', '72_1', '72_2', '72_3']

    EC_H_lib_path = os.path.join(result_root_path, 'EC_H', 'EC_H_annotated.csv')
    EC_H_samples = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18', '19', '20', '21']

    UPS_M_lib_path = os.path.join(result_root_path, 'UPS_M', 'UPS_M_annotated.csv')
    UPS_M_samples = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']

    UPS_Y_lib_path = os.path.join(result_root_path, 'UPS_Y', 'UPS_Y_annotated.csv')
    UPS_Y_samples = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']

    Benchmark_FC_lib_path = os.path.join(result_root_path, 'Benchmark_FC', 'Benchmark_FC_annotated.csv')
    Benchmark_FC_samples = ['10_R1', '10_R2', '10_R3', '15_R1', '15_R2', '15_R3', '20_R1', '20_R2', '20_R3', '25_R1',
                            '25_R2', '25_R3']


    MTBLS736_TripleTOF_6600_lib_matrix = load_lib(MTBLS736_TripleTOF_6600_lib_path, MTBLS736_TripleTOF_6600_samples)
    MTBLS733_QE_HF_lib_matrix = load_lib(MTBLS733_QE_HF_lib_path, MTBLS733_QE_HF_samples)
    MTBLS3038_NEG_lib_matrix = load_lib(MTBLS3038_NEG_lib_path, MTBLS3038_NEG_samples)
    MTBLS3038_POS_lib_matrix = load_lib(MTBLS3038_POS_lib_path, MTBLS3038_POS_samples)
    MTBLS5430_Lip_NEG_lib_matrix = load_lib(MTBLS5430_Lip_NEG_lib_path, MTBLS5430_Lip_NEG_samples)
    MTBLS5430_Lip_POS_lib_matrix = load_lib(MTBLS5430_Lip_POS_lib_path, MTBLS5430_Lip_POS_samples)
    MTBLS5430_Metabo_NEG_lib_matrix = load_lib(MTBLS5430_Metabo_NEG_lib_path, MTBLS5430_Metabo_NEG_samples)
    MTBLS5430_Metabo_POS_lib_matrix = load_lib(MTBLS5430_Metabo_POS_lib_path, MTBLS5430_Metabo_POS_samples)
    AT_lib_matrix = load_lib(AT_lib_path, AT_samples)
    EC_H_lib_matrix = load_lib(EC_H_lib_path, EC_H_samples)
    UPS_M_lib_matrix = load_lib(UPS_M_lib_path, UPS_M_samples)
    UPS_Y_lib_matrix = load_lib(UPS_Y_lib_path, UPS_Y_samples)
    Benchmark_FC_lib_matrix = load_lib(Benchmark_FC_lib_path, Benchmark_FC_samples)


    def load_eval(feature_source, align_method, lib_matrix, aligned_result_file_name, dataset_name, sample_names,
                  mz_tolerance=0.0001, rt_tolerance=0.0001):
        try:
            if "_" in align_method:
                temp_align_method = align_method.split('_', 1)[0]
            else:
                temp_align_method = align_method
            result_paths = [glob.glob(os.path.join(result_root_path, dataset_name, feature_source, name + '.csv'))[0]
                            for name in sample_names]
            result_matrix = load_features(result_paths, separator=',', skip_line=0, mz_col_idx=0, rt_col_idx=1,
                                          area_col_idx=2)
            aligned_result_path = os.path.join(result_root_path, dataset_name + "_results_" + feature_source,
                                               temp_align_method, aligned_result_file_name)
            aligned_matrix = load_aligned_result(aligned_result_path)[:, 4:]
            eval(lib_matrix, result_matrix, aligned_matrix, mz_tolerance, rt_tolerance)
            print(dataset_name, feature_source, align_method, "finished")
        except Exception:
            pass




    # dataset_names = ["MTBLS733_QE_HF", "MTBLS736_TripleTOF_6600", "MTBLS3038_NEG", "MTBLS3038_POS", "MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS", "MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS", "AT", "EC_H", "Benchmark_FC","UPS_M", "UPS_Y"] #"MTBLS733_QE_HF", "MTBLS736_TripleTOF_6600", "MTBLS3038_NEG", "MTBLS3038_POS", "MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS", "MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS", "AT", "EC_H", "Benchmark_FC", "UPS_M", "UPS_Y"
    dataset_names = ["MTBLS5430_Lip_POS"]
    feature_sources = ["metapro"]#"metapro", "mzmine2", "openms", "xcms", "AntDAS", "dinosaur", "maxquant"
    methods = ["trend-aligner"]
    #"trend-aligner", "deeprtalign", "mzmine2_join", "mzmine2_ransac", "openms", "M2S", "xcms_group", "xcms_obiwarp","AntDAS_group-based","AntDAS_coarse+precise","AntDAS_direct","AntDAS_coarse+nearest"
    for dataset_name in dataset_names:
        for feature_source in feature_sources:
            for method in methods:
                if feature_source == "metapro":
                    mz_tolerance = 0.0001
                    rt_tolerance = 0.0001
                else:
                    mz_tolerance = 0.01
                    rt_tolerance = 0.1
                aligned_matrix = dataset_name + "_aligned_" + method + ".csv"
                samples = locals()[dataset_name + "_samples"]
                lib_matrix = locals()[dataset_name + "_lib_matrix"]
                load_eval(feature_source, method, lib_matrix, aligned_matrix, dataset_name, samples, mz_tolerance, rt_tolerance)
        print(dataset_name, "finished")


