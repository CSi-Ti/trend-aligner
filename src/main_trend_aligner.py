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
import argparse
import time
import copy
import numpy as np
from src.utils.params import FeatureListReadingParams, CoarseAlignmentParams, FineMatchingParams
from src.utils import file_utils
from src.coarse_rt_alignment import lowess_align
from src.fine_feature_matching import estimate_pu, estimate_qi_and_match


class TrendAligner:
    def __init__(self, feature_reading_params, coarse_alignment_params, fine_matching_params, save_name=None, save_path=None, plot=False):
        self.feature_reading_params = feature_reading_params
        self.coarse_alignment_params = coarse_alignment_params
        self.fine_matching_params = fine_matching_params

        if save_name is None:
            self.save_name = str(time.strftime('%Y%m%d_%H%M%S', time.localtime())) + '_alignment_results.csv'
        else:
            self.save_name = save_name
        if save_path is None:
            self.save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'alignment_results')
        else:
            self.save_path = save_path
        self.plot = plot
        file_utils.save_params(self.save_path, feature_reading_params, coarse_alignment_params,
                               fine_matching_params)

        print('Trend-Aligner started...')
        print('Alignment results will be saved in %s' % self.save_path)

    def do_align(self):
        global_start_time = time.time()

        # 1. Load result data
        print('Finding result files from result folder... ')
        feature_lists, file_names = \
            file_utils.load_feature_lists(self.feature_reading_params.feature_list_folder_path,
                                          self.feature_reading_params.skip_line, None,
                                          self.feature_reading_params.mz_col_idx,
                                          self.feature_reading_params.rt_col_idx,
                                          self.feature_reading_params.area_col_idx)
        sorted_indices = np.argsort(file_names)
        feature_lists = feature_lists[sorted_indices]
        file_names = file_names[sorted_indices]
        feature_lists = [data[np.argsort(data[:, 0])] for data in feature_lists]
        feature_lists = [np.hstack((data, np.full((data.shape[0], 1), i))) for i, data in enumerate(feature_lists)]
        print('\tFound and loaded %d result files. \t%.1fs' % (len(feature_lists), time.time() - global_start_time))
        print('\t' + str(file_names))

        # 2. Coarse registration
        start_time = time.time()
        print('Performing coarse registration...')
        print('\tCalculating lowess functions from result data...')
        warp_funcs, warp_data = lowess_align(feature_lists, self.coarse_alignment_params, self.save_path, self.plot)
        print('\tApplying lowess functions to result data...')
        coarse_aligned_feature_lists = copy.deepcopy(feature_lists)
        for i in range(len(feature_lists)):
            if warp_funcs[i] is not None:
                coarse_aligned_feature_lists[i][:, 1] = warp_funcs[i](coarse_aligned_feature_lists[i][:, 1])
        print('\tCoarse registration finished. \t%.1fs' % (time.time() - start_time))

        # 3. Fine assignment
        start_time = time.time()
        print('Analyzing RT drift trend...')
        print('\tCalculating Trend-Aligner model parameters...')
        pu = estimate_pu(coarse_aligned_feature_lists, self.fine_matching_params.beam_mz_tol,
                         self.fine_matching_params.beam_rt_tol, self.fine_matching_params.use_ppm,
                         self.fine_matching_params.beam_width, self.save_path, self.plot)
        print('\tMatching features by predicted RTs...')
        match_result = estimate_qi_and_match(feature_lists, coarse_aligned_feature_lists, pu,
                                             self.fine_matching_params.match_mz_tol,
                                             self.fine_matching_params.match_rt_tol,
                                             self.fine_matching_params.max_rt_tol, self.fine_matching_params.use_ppm,
                                             self.fine_matching_params.mz_factor, self.fine_matching_params.rt_factor,
                                             self.fine_matching_params.area_factor)
        aligned_matrix = np.ones((len(match_result), 3 * len(feature_lists))) * -1
        for i, match in enumerate(match_result):
            for feature in match:
                feature_id = int(feature[-1])
                aligned_matrix[i][3 * feature_id] = feature[0]
                aligned_matrix[i][3 * feature_id + 1] = feature[1]
                aligned_matrix[i][3 * feature_id + 2] = feature[2]
        print('\tMatched %d consensus features. \t%.1fs' % (len(aligned_matrix), time.time() - start_time))

        # 5. Save results to disk
        start_time = time.time()
        print('Saving results to disk...')
        result_file_path = file_utils.save_alignment_results(aligned_matrix, file_names, self.save_path, self.save_name)
        print('\tResult file saved in ' + result_file_path + ' \t%.1fs' % (time.time() - start_time))
        print('Finished. Trend-Aligner aligned %d files in %.1fs.' % (len(file_names), time.time() - global_start_time))


if __name__ == '__main__':
    def press_or_float(value):
        try:
            return float(value)
        except ValueError:
            if value == 'tPRESS':
                return value
            raise argparse.ArgumentTypeError(f"'{value}' must be float or 'tPRESS'")

    parser = argparse.ArgumentParser(description='Trend-Aligner parameters')

    # Result file reading params
    parser.add_argument('--feature_list_folder', type=str, help='Path to feature list folder', required=True)
    parser.add_argument('--skip_line', type=int, help='Number of header rows ', required=True)
    parser.add_argument('--mz_col_num', type=int, help='M/z column number', required=True)
    parser.add_argument('--rt_col_num', type=int, help='RT column number', required=True)
    parser.add_argument('--area_col_num', type=int, help='Area column number', required=True)

    # Coarse alignment parameters
    parser.add_argument('--centric_idx', type=float, help='Centric run index in RANSAC alignment', required=False, default=-1)
    parser.add_argument('--mz_tolerance', type=float, help='M/z tolerance in coarse alignment', required=False, default=5)
    parser.add_argument('--rt_tolerance', type=float, help='RT tolerance in coarse alignment', required=False, default=0.5)
    parser.add_argument('--use_ppm', type=float, help='Use ppm m/z tolerance', required=False, default=True)
    parser.add_argument('--from_rt', type=float, help='RT start in coarse alignment', required=False, default=0)
    parser.add_argument('--to_rt', type=float, help='RT end in coarse alignment', required=False, default=float('inf'))
    parser.add_argument('--frac', type=press_or_float, help='lowess bandwidth in coarse alignment', required=False, default='tPRESS')

    # Fine matching parameters
    parser.add_argument('--beam_mz_tol', type=float, help='M/z tolerance in beam search', required=False, default=None)
    parser.add_argument('--beam_rt_tol', type=float, help='RT tolerance in beam search', required=False, default=None)
    parser.add_argument('--match_mz_tol', type=float, help='M/z tolerance in matching between neighbor runs', required=True)
    parser.add_argument('--match_rt_tol', type=float, help='RT tolerance in matching between neighbor runs', required=True)
    parser.add_argument('--max_rt_tol', type=float, help='Max RT tolerance in matching', required=True)
    parser.add_argument('--use_ppm', type=bool, help='Use ppm m/z tolerance', required=False, default=False)
    parser.add_argument('--beam_width', type=int, help='Beam search neighbors', required=False, default=1)
    parser.add_argument('--mz_factor', type=float, help='M/z factor in nearest matching', required=False, default=1)
    parser.add_argument('--rt_factor', type=float, help='RT factor in nearest matching', required=False, default=1)
    parser.add_argument('--area_factor', type=float, help='Area factor in nearest matching', required=False, default=0)

    # other parameters
    parser.add_argument('--save_name', type=str, help='Alignment result name (including file extension, e.g., .csv)', required=False, default=None)
    parser.add_argument('--save_path', type=str, help='Directory path where the alignment result will be saved', required=False, default=None)
    parser.add_argument('--plot', type=bool, help='Whether to generate plots (True or False)', required=False, default=False)


    args = parser.parse_args()

    result_file_reading_params = FeatureListReadingParams(feature_list_folder_path=args.feature_list_folder,
                                                          skip_line=args.skip_line, mz_col_num=args.mz_col_num,
                                                          rt_col_num=args.rt_col_num, area_col_num=args.area_col_num)

    coarse_alignment_params = CoarseAlignmentParams(centric_idx=args.centric_idx, mz_tolerance=args.mz_tolerance,
                                                    use_ppm=args.use_ppm, rt_tolerance=args.rt_tolerance,
                                                    frac=args.frac, from_rt=args.from_rt, to_rt=args.to_rt)

    fine_matching_params = FineMatchingParams(match_mz_tol=args.match_mz_tol, match_rt_tol=args.match_rt_tol,
                                              max_rt_tol=args.max_rt_tol, beam_mz_tol=args.beam_mz_tol,
                                              beam_rt_tol=args.beam_rt_tol, use_ppm=args.use_ppm,
                                              beam_width=args.beam_width, mz_factor=args.mz_factor,
                                              rt_factor=args.rt_factor, area_factor=args.area_factor)

    trend_aligner = TrendAligner(result_file_reading_params, coarse_alignment_params, fine_matching_params, args.save_name, args.save_path, args.plot)
    trend_aligner.do_align()
