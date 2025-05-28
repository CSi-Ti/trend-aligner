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
from src.main_trend_aligner import TrendAligner
from src.utils.params import FeatureListReadingParams, CoarseAlignmentParams, FineMatchingParams

root_path = 'E:\\workspace'

def align(dataset_name, feature_source, skip_line, mz_col_num, rt_col_num, area_col_num, mz_tolerance, use_ppm, centric_idx, rt_tolerance, frac,
          beam_mz_tol, beam_rt_tol, match_mz_tol, match_rt_tol, max_rt_tol, save_path, save_name, plot):
    try:
        feature_folder_path = os.path.join(root_path, dataset_name, feature_source)
        feature_reading_params = FeatureListReadingParams(feature_folder_path, skip_line, mz_col_num, rt_col_num, area_col_num)
        coarse_alignment_params = CoarseAlignmentParams(mz_tolerance=mz_tolerance, use_ppm=use_ppm, centric_idx=centric_idx,
                                                              rt_tolerance=rt_tolerance, frac=frac)
        fine_matching_params = FineMatchingParams(beam_mz_tol=beam_mz_tol, beam_rt_tol=beam_rt_tol,
                                                  match_mz_tol=match_mz_tol, match_rt_tol=match_rt_tol,
                                                  max_rt_tol=max_rt_tol)
        trend_aligner = TrendAligner(feature_reading_params, coarse_alignment_params, fine_matching_params, save_name, save_path, plot)

        trend_aligner.do_align()
    except Exception:
        pass


CoarseParams = {
    "MTBLS733_QE_HF": {"mz_tolerance": 0.005,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 0.3,"frac": 'tPRESS'
    },
    "MTBLS736_TripleTOF_6600": {"mz_tolerance": 0.005, "use_ppm": False, "centric_idx": 0, "rt_tolerance": 1, "frac": 'tPRESS'
    },
    "MTBLS3038_NEG": {"mz_tolerance": 0.03, "use_ppm": False, "centric_idx": 0, "rt_tolerance": 0.1, "frac": 'tPRESS'
    },
    "MTBLS3038_POS": {"mz_tolerance": 0.005,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 0.3,"frac": 'tPRESS'
    },
    "MTBLS5430_Lip_NEG": {"mz_tolerance": 0.01,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 0.5,"frac": 'tPRESS'
    },
    "MTBLS5430_Lip_POS": {"mz_tolerance": 0.01,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 0.3,"frac": 'tPRESS'
    },
    "MTBLS5430_Metabo_NEG": {"mz_tolerance": 0.005,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 0.3,"frac": 'tPRESS'
    },
    "MTBLS5430_Metabo_POS": {"mz_tolerance": 0.01,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 0.5,"frac": 'tPRESS'
    },
    "AT": {"mz_tolerance": 0.01, "use_ppm": False, "centric_idx": 0, "rt_tolerance": 2, "frac": 'tPRESS'
    },
    "EC_H": {"mz_tolerance": 0.01,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 1, "frac": 'tPRESS'
    },
    "Benchmark_FC": {"mz_tolerance": 0.03,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 1,"frac": 'tPRESS'
    },
    "UPS_M": {"mz_tolerance": 0.01, "use_ppm": False, "centric_idx": 0, "rt_tolerance": 2, "frac": 'tPRESS'
    },
    "UPS_Y": {"mz_tolerance": 0.01,"use_ppm": False,"centric_idx": 0,"rt_tolerance": 4,"frac": 'tPRESS'
    }
}


FineParams = {
    "MTBLS733_QE_HF": {"beam_mz_tol": 0.005,"beam_rt_tol": 0.2,"match_mz_tol": 0.005,"match_rt_tol": 0.2,"max_rt_tol": 0.8
    },
    "MTBLS736_TripleTOF_6600": {"beam_mz_tol": 0.005,"beam_rt_tol": 0.2,"match_mz_tol": 0.02,"match_rt_tol": 0.2,"max_rt_tol": 1
    },
    "MTBLS3038_NEG": {"beam_mz_tol": 0.02, "beam_rt_tol": 0.2, "match_mz_tol": 0.03, "match_rt_tol": 0.2, "max_rt_tol": 0.5
    },
    "MTBLS3038_POS": {"beam_mz_tol": 0.005,"beam_rt_tol": 0.1,"match_mz_tol": 0.02,"match_rt_tol": 0.1,"max_rt_tol": 0.5
    },
    "MTBLS5430_Lip_NEG": {"beam_mz_tol": 0.005,"beam_rt_tol": 0.05,"match_mz_tol": 0.02,"match_rt_tol": 0.2,"max_rt_tol": 0.4
    },
    "MTBLS5430_Lip_POS": {"beam_mz_tol": 0.01, "beam_rt_tol": 0.2, "match_mz_tol": 0.02, "match_rt_tol": 0.4, "max_rt_tol": 0.8
    },
    "MTBLS5430_Metabo_NEG": {"beam_mz_tol": 0.001, "beam_rt_tol": 0.1, "match_mz_tol": 0.01, "match_rt_tol": 0.2, "max_rt_tol": 0.5
    },
    "MTBLS5430_Metabo_POS": {"beam_mz_tol": 0.005,"beam_rt_tol": 0.05,"match_mz_tol": 0.01,"match_rt_tol": 0.2,"max_rt_tol": 0.5
    },
    "AT": {"beam_mz_tol": 0.01, "beam_rt_tol": 0.5, "match_mz_tol": 0.02, "match_rt_tol": 1, "max_rt_tol": 2
    },
    "EC_H": {"beam_mz_tol": 0.01,"beam_rt_tol": 0.5,"match_mz_tol": 0.01,"match_rt_tol": 0.5,"max_rt_tol": 1
    },
    "Benchmark_FC": {"beam_mz_tol": 0.01, "beam_rt_tol": 0.3, "match_mz_tol": 0.01, "match_rt_tol": 0.3,"max_rt_tol": 1
    },
    "UPS_M": {"beam_mz_tol": 0.01, "beam_rt_tol": 0.5, "match_mz_tol": 0.01, "match_rt_tol": 0.5, "max_rt_tol": 1
    },
    "UPS_Y": {"beam_mz_tol": 0.01, "beam_rt_tol": 1, "match_mz_tol": 0.01, "match_rt_tol": 1, "max_rt_tol": 2
    }
}

# dataset_names = ["MTBLS733_QE_HF", "MTBLS736_TripleTOF_6600", "MTBLS3038_NEG", "MTBLS3038_POS", "MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS", "MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS", "AT", "EC_H", "Benchmark_FC","UPS_M", "UPS_Y"] #"MTBLS733_QE_HF", "MTBLS736_TripleTOF_6600", "MTBLS3038_NEG", "MTBLS3038_POS", "MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS", "MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS", "AT", "EC_H", "Benchmark_FC", "UPS_M", "UPS_Y"
dataset_names = ["MTBLS736_TripleTOF_6600"]
feature_sources = ["metapro"]#"metapro", "mzmine2", "openms", "xcms", "AntDAS", "dinosaur", "maxquant"

for dataset_name in dataset_names:
    dataset_coarseparams = CoarseParams[dataset_name]
    dataset_fineparams = FineParams[dataset_name]
    for feature_source in feature_sources:
        save_path = os.path.join(root_path, dataset_name + "_results_" + feature_source, "trend-aligner")
        save_name = dataset_name + "_aligned_trend-aligner.csv"
        align(
            dataset_name,
            feature_source,
            skip_line=0,
            mz_col_num=1,
            rt_col_num=2,
            area_col_num=3,
            mz_tolerance=dataset_coarseparams["mz_tolerance"],
            use_ppm=dataset_coarseparams["use_ppm"],
            centric_idx=dataset_coarseparams["centric_idx"],
            rt_tolerance=dataset_coarseparams["rt_tolerance"],
            frac=dataset_coarseparams["frac"],
            beam_mz_tol=dataset_fineparams["beam_mz_tol"],
            beam_rt_tol=dataset_fineparams["beam_rt_tol"],
            match_mz_tol=dataset_fineparams["match_mz_tol"],
            match_rt_tol=dataset_fineparams["match_rt_tol"],
            max_rt_tol=dataset_fineparams["max_rt_tol"],
            save_path=save_path,
            save_name=save_name,
            plot=False
        )

