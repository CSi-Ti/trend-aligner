# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren, Etienne Caron and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
from scipy.stats import gaussian_kde
from src.utils import plot_utils


def _find_matches(features, target_mz, target_rt, mz_tolerance, rt_tolerance):
    # IMPORTANT: feature_list must be sorted before by mz
    mz_start = target_mz - mz_tolerance
    mz_end = target_mz + mz_tolerance
    rt_start = target_rt - rt_tolerance
    rt_end = target_rt + rt_tolerance

    mz_left_idx = np.searchsorted(features[:, 0], mz_start, side='left')
    mz_right_idx = np.searchsorted(features[:, 0], mz_end, side='right')

    match_state = (features[mz_left_idx: mz_right_idx, 1] >= rt_start) * (features[mz_left_idx: mz_right_idx, 1] <= rt_end)
    match_indices = np.arange(mz_left_idx, mz_right_idx)[match_state]
    return match_indices


def _find_neighbors(target_list, target_idxes, feature_list, mz_tolerance, rt_tolerance, use_ppm, max_neighbors):
    # IMPORTANT: target_list, feature_list must be sorted before by mz
    match_map = {}
    for i in target_idxes:
        target = target_list[i]
        tmp_mz_tolerance = target[0] * mz_tolerance * 1e-6 if use_ppm else mz_tolerance
        match_indices = _find_matches(feature_list, target[0], target[1], tmp_mz_tolerance, rt_tolerance)

        if max_neighbors is None or len(match_indices) <= max_neighbors:
            match_map[i] = match_indices
        else:
            match_indices = match_indices[np.argsort(-feature_list[match_indices, 2])]
            match_map[i] = match_indices[:max_neighbors]

    return match_map


def beam_search(feature_list, mz_tolerance, rt_tolerance, use_ppm, beam_width):
    target_idxes = np.arange(len(feature_list[0]))
    match_list = [[i] for i in range(len(feature_list[0]))]
    for i in range(len(feature_list) - 1):
        tmp_target_idxes = set()
        tmp_match_list = []
        neighbor_map = _find_neighbors(feature_list[i], target_idxes, feature_list[i + 1], mz_tolerance, rt_tolerance,
                                       use_ppm, max_neighbors=beam_width)
        for match in match_list:
            for idx in neighbor_map[match[-1]]:
                tmp_match_list.append(match + [idx])
                tmp_target_idxes.add(idx)
        target_idxes = tmp_target_idxes
        match_list = tmp_match_list
    return np.array(match_list)


def estimate_pu(feature_lists, beam_mz_tol, beam_rt_tol, use_ppm, beam_width, save_path, plot):
    match_list = beam_search(feature_lists, beam_mz_tol, beam_rt_tol, use_ppm, beam_width)

    rt_matrix = np.zeros((len(match_list), len(feature_lists)))
    for i in range(len(feature_lists)):
        rt_matrix[:, i] = feature_lists[i][match_list[:, i], 1]

    rt_dev_matrix = rt_matrix - rt_matrix[:, 0:1]
    normed_rt_dev_matrix = rt_dev_matrix / rt_dev_matrix[
        np.arange(len(rt_dev_matrix)), np.argmax(np.abs(rt_dev_matrix), axis=-1)].reshape(-1, 1)
    normed_rt_dev_matrix = normed_rt_dev_matrix[~np.isnan(normed_rt_dev_matrix).any(axis=1)]
    normed_rt_dev_matrix = normed_rt_dev_matrix[~np.isinf(normed_rt_dev_matrix).any(axis=1)]

    kde = gaussian_kde(normed_rt_dev_matrix[:, 1:].T, bw_method='scott')  #silverman, scott
    density = kde(normed_rt_dev_matrix[:, 1:].T)
    max_density_idx = np.argmax(density)
    pu = normed_rt_dev_matrix[max_density_idx]
    if np.max(np.abs(rt_dev_matrix[max_density_idx])) < 0.1:
        pu = np.zeros(pu.shape)

    if plot == True:
        plot_utils.plt_rt_dev(rt_dev_matrix, save_path)
        plot_utils.plt_pu(normed_rt_dev_matrix, pu, save_path)

    return pu



def estimate_qi_and_match(feature_list_raw, feature_list_calibrated, pu, match_mz_tol, match_rt_tol, max_rt_tol,
                          use_ppm, mz_factor, rt_factor, area_factor):
    feature_list = []
    for features in feature_list_raw:
        feature_list.append(features.copy())
    match_result = []
    for f in range(len(feature_list_calibrated)):
        matches = []
        # if f == 0:
        #     import os
        #     import pandas as pd
        #     lib = pd.read_csv(os.path.join("E:\workspace\MTBLS736_TripleTOF_6600\MTBLS736_TripleTOF_6600_annotated.csv"))
        #     error = {}
        # if f == 1:
        #     import matplotlib.pyplot as plt
        #     df_error = pd.DataFrame(error).T
        #     plt.figure(figsize=(10, 8))
        #     for column in df_error.columns:
        #         plt.scatter(df_error.index, df_error[column], label=column, s=5)
        #     plt.rcParams['savefig.dpi'] = 600
        #
        #     plt.title("MTBLS736_TripleTOF_6600", fontsize=25, fontweight='regular', fontfamily='Arial', pad=20)
        #     plt.xlabel("RT(min)", fontsize=25, fontweight='regular', fontfamily='Arial', labelpad=5)
        #     plt.ylabel("RT shift(min)", fontsize=25, fontweight='regular', fontfamily='Arial', labelpad=5)
        #     plt.ylim(-0.6, 0.48)
        #     plt.tick_params(axis='both', which='major', labelsize=20)  # 设置主刻度标签大小
        #     plt.tick_params(axis='both', which='minor', labelsize=20)  # 设置次刻度标签大小
        #     plt.subplots_adjust(left=0.2)
        #
        #     # plt.savefig(os.path.join("E:\workspace_plot\performance", dataset_name + ".png"), bbox_inches='tight')
        #     plt.show()
        for i in range(len(feature_list_calibrated[f])):
            target = feature_list_calibrated[f][i]
            matched_normed_drifts = []
            matched_run_indices = set()
            match_idx_map = {}
            tmp_mz_tol = target[0] * match_mz_tol * 1e-6 if use_ppm else match_mz_tol
            # Find potential matches in different runs
            for j in range(f + 1, len(feature_list_calibrated)):
                tmp_rt_tol = (max_rt_tol - match_rt_tol) * abs(pu[j] - pu[f]) + match_rt_tol
                matched_indices = _find_matches(feature_list_calibrated[j], target[0], target[1], tmp_mz_tol, tmp_rt_tol)
                match_idx_map[j] = matched_indices
                matched_rts = feature_list_calibrated[j][matched_indices, 1]
                if len(matched_rts) > 0:
                    matched_run_indices.add(j)
                if pu[j] == pu[f]:
                    matched_normed_drifts.append(0)
                else:
                    normed_drifts = (matched_rts - target[1]) / (pu[j] - pu[f])
                    matched_normed_drifts.extend(normed_drifts)
            if len(matched_run_indices) == 0:  # TODO
                continue

            # Estimate QI
            if min(matched_normed_drifts) == max(matched_normed_drifts):
                qi = matched_normed_drifts[0]
            else:
                peak_density_estimation = gaussian_kde(matched_normed_drifts, bw_method=0.1) #silverman, scott
                peak_density = peak_density_estimation(matched_normed_drifts)
                max_density_idx = np.argmax(peak_density)
                qi = matched_normed_drifts[max_density_idx]

            # Find the nearest match by predicted RT
            pred_rts = target[1] + (pu - pu[f]) * qi
            # if f == 0:
            #     gloabal_residual = []
            #     for m in range(len(warp_funcs)):
            #         if m == 0:
            #             residual = 0
            #         else:
            #             residual = warp_funcs[m](target[1]) - target[1]
            #         gloabal_residual.append(residual)
            #     pred_rts_plot = pred_rts - np.array(gloabal_residual)
            #     raw_target = feature_list[f][i]
            #     matched_index = None
            #     for idx, row in lib.iloc[:, [5, 6, 7]].iterrows():
            #         if raw_target[0]-0.01 < row[0] < raw_target[0]+0.01:
            #             if raw_target[1]-0.1 < row[1] < raw_target[1]+0.1:
            #                 if abs(raw_target[2]-row[2])/row[2] < 1e-6:
            #                     matched_index = idx
            #                     break
            #     if matched_index is not None:
            #         row = lib.iloc[matched_index]
            #         rt_columns = [col for col in lib.columns if col.endswith('_Rt')]
            #         rt_condition = [col for col in lib.columns if col.endswith('_ManStatus')]
            #         rt_values = row[rt_columns].values
            #         condition_values = row[rt_condition].values
            #         match_diff = np.where(condition_values == "Success", rt_values - pred_rts_plot, 0)
            #         new_key = raw_target[1]
            #         while new_key in error:
            #             new_key += 1e-4
            #         if len(match_diff) > 0:
            #             error[new_key] = match_diff
            match = np.ones(len(feature_list_calibrated)) * -1
            match[f] = i
            for j in range(f + 1, len(feature_list_calibrated)):
                match_idxes = match_idx_map[j]
                if len(match_idxes) == 0:
                    continue
                features = feature_list_calibrated[j][match_idxes]
                min_dist = 20
                nearest_idx = -1
                for k, feature in enumerate(features):
                    rt_dev = abs(pred_rts[j] - feature[1])
                    if rt_dev > match_rt_tol:
                        continue

                    dist = 0
                    if mz_factor > 0: dist += mz_factor * np.power((target[0] - feature[0]) / tmp_mz_tol, 2)
                    if rt_factor > 0: dist += rt_factor * np.power(rt_dev / match_rt_tol, 2)
                    if area_factor > 0: dist += area_factor * np.power(np.log10(target[2] + 1) - np.log10(feature[2] + 1), 2)
                    dist /= mz_factor + rt_factor + area_factor
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = k
                if nearest_idx != -1:
                    match[j] = match_idxes[nearest_idx]
            matches.append(match)

        if len(matches) == 0:
            continue
        matches = np.array(matches, dtype=np.int)

        # Convert matched indices to features
        for match in matches:
            matched_features = []
            for j in np.where(match >= 0)[0]:
                matched_features.append(feature_list[j][match[j]])
            match_result.append(matched_features)

        for i in range(f + 1, len(matches[0])):
            del_idxes = np.sort(np.unique(matches[:, i]))
            if del_idxes[0] == -1:
                del_idxes = del_idxes[1:]
            feature_list_calibrated[i] = np.delete(feature_list_calibrated[i], del_idxes, axis=0)
            feature_list[i] = np.delete(feature_list[i], del_idxes, axis=0)

    return match_result