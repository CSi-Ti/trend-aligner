# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

class FeatureListReadingParams:
    def __init__(self, feature_list_folder_path, skip_line, mz_col_num, rt_col_num, area_col_num):

        self.feature_list_folder_path = feature_list_folder_path
        self.skip_line = skip_line
        self.mz_col_idx = mz_col_num - 1
        self.rt_col_idx = rt_col_num - 1
        self.area_col_idx = area_col_num - 1


class CoarseAlignmentParams:
    def __init__(self, centric_idx=0, from_rt=0, to_rt=float('inf'), mz_tolerance=5, use_ppm=True,
                 rt_tolerance=0.5, frac='tPRESS'):

        self.centric_idx = centric_idx
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.frac = frac
        self.use_ppm = use_ppm
        self.from_rt = from_rt
        self.to_rt = to_rt

        # Parameter verification
        assert isinstance(self.centric_idx, int) and centric_idx >= 0
        assert isinstance(self.mz_tolerance, float) or isinstance(self.mz_tolerance, int) and mz_tolerance > 0
        assert isinstance(self.rt_tolerance, float) or isinstance(self.rt_tolerance, int) and rt_tolerance > 0
        assert isinstance(self.use_ppm, bool)
        assert isinstance(self.from_rt, float) or isinstance(self.from_rt, int)
        assert isinstance(self.to_rt, float) or isinstance(self.to_rt, int)
        if self.frac is not 'tPRESS':
            assert isinstance(self.frac, float) and 0 < self.frac <= 1


class FineMatchingParams:
    def __init__(self, beam_mz_tol, beam_rt_tol, match_mz_tol, match_rt_tol, max_rt_tol,
                 use_ppm=False, beam_width=1, mz_factor=1, rt_factor=1, area_factor=0):
        self.beam_mz_tol = beam_mz_tol if beam_mz_tol is not None else match_mz_tol / 4.0
        self.beam_rt_tol = beam_rt_tol if beam_rt_tol is not None else match_rt_tol
        self.match_mz_tol = match_mz_tol
        self.match_rt_tol = match_rt_tol
        self.max_rt_tol = max_rt_tol
        self.use_ppm = use_ppm
        self.beam_width = beam_width
        self.mz_factor = mz_factor
        self.rt_factor = rt_factor
        self.area_factor = area_factor

        # Parameter verification
        assert (isinstance(match_mz_tol, (float, int)) and match_mz_tol > 0)
        assert (isinstance(match_rt_tol, (float, int)) and match_rt_tol > 0)
        assert (isinstance(max_rt_tol, (float, int)) and max_rt_tol > 0)
        assert (isinstance(beam_mz_tol, (float, int)) and beam_mz_tol > 0)
        assert (isinstance(beam_rt_tol, (float, int)) and beam_rt_tol > 0)
        assert isinstance(use_ppm, bool)
        assert (match_mz_tol > 1 and beam_mz_tol > 1 and use_ppm) or (match_mz_tol < 1 and beam_mz_tol < 1 and not use_ppm)
        assert isinstance(beam_width, int) and beam_width > 0
        assert (isinstance(rt_factor, (float, int)) and rt_factor >= 0)
        assert (isinstance(mz_factor, (float, int)) and mz_factor >= 0)
        assert (isinstance(area_factor, (float, int)) and area_factor >= 0)
        assert mz_factor + rt_factor + area_factor > 0, 'Wrong factors. Sum of factors cannot be 0.'
