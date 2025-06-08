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
from src.main_trend_aligner import TrendAligner
from src.utils.params import FeatureListReadingParams, CoarseAlignmentParams, FineMatchingParams
from experiment.tools.feature_converter.FeatureConverter import FeatureConverter
import glob

root_path = os.path.dirname(os.path.abspath(__file__))

def align(feature_reading_params, coarse_alignment_params, fine_matching_params, save_name=None, save_path=None, plot=False):
    trend_aligner = TrendAligner(feature_reading_params, coarse_alignment_params, fine_matching_params, save_name, save_path, plot)
    trend_aligner.do_align()


#demo
example_featurelist_path = os.path.join(root_path, "metapro_example")
save_name = "MTBLS736_TripleTOF_6600_aligned_trend-aligner.csv"
save_path = os.path.join(root_path, "metapro_result")

feature_reading_params = FeatureListReadingParams(feature_list_folder_path=example_featurelist_path, skip_line=0, mz_col_num=1, rt_col_num=2, area_col_num=3)
coarse_alignment_params = CoarseAlignmentParams(mz_tolerance=0.005, use_ppm=False, centric_idx=0, rt_tolerance=1, frac='tPRESS')
fine_matching_params = FineMatchingParams(beam_mz_tol=0.005, beam_rt_tol=0.2, match_mz_tol=0.02, match_rt_tol=0.2, max_rt_tol=1, use_ppm=False)

align(feature_reading_params, coarse_alignment_params, fine_matching_params, save_name=save_name, save_path=save_path, plot=True)



#demo(openms_featurelist)
example_featurelist_openms_paths = glob.glob(os.path.join(root_path, "openms_example", "*.featureXML"))
for path in example_featurelist_openms_paths:
    converter = FeatureConverter(path=path, software="openms", save_folder=os.path.join(root_path, "openms_example_converted"), suffix=".featureXML")
    converter.convert()
converted_featurelist_openms_path = os.path.join(root_path, "openms_example_converted")
save_name = "EC_H_aligned_trend-aligner.csv"
save_path = os.path.join(root_path, "openms_result")

feature_reading_params = FeatureListReadingParams(feature_list_folder_path=converted_featurelist_openms_path, skip_line=0, mz_col_num=1, rt_col_num=2, area_col_num=3)
coarse_alignment_params = CoarseAlignmentParams(mz_tolerance=0.01, use_ppm=False, centric_idx=0, rt_tolerance=1, frac='tPRESS')
fine_matching_params = FineMatchingParams(beam_mz_tol=0.01, beam_rt_tol=0.5, match_mz_tol=0.01, match_rt_tol=0.5, max_rt_tol=1, use_ppm=False)

align(feature_reading_params, coarse_alignment_params, fine_matching_params, save_name=save_name, save_path=save_path, plot=True)