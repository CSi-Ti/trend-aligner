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
import matplotlib.pyplot as plt
from datetime import datetime

def plt_lowess_fitting_results(candidate_from_rts, candidate_to_rts, result_lowess, lowess_function, save_path):
    plt.rcParams['savefig.dpi'] = 1200
    save_path = os.path.join(save_path, 'lowess_fitting_results')
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    file_name = f"{timestamp}.png"
    save_path = os.path.join(save_path, file_name)

    plt.scatter(candidate_from_rts, candidate_to_rts - candidate_from_rts, color='blue', alpha=0.5, s=4)
    plt.scatter(result_lowess[:, 0], result_lowess[:, 1] - result_lowess[:, 0], color='red', label='LOWESS', alpha=0.5, s=4)
    x_values = np.linspace(min(candidate_from_rts), max(candidate_from_rts), int(max(candidate_from_rts) - min(candidate_from_rts))).reshape(-1, 1)
    y_values = lowess_function(x_values)
    plt.plot(x_values, y_values - x_values.reshape(-1, 1), color='green', label='LOWESS Fitting Function', linewidth=1)

    plt.xlabel('RT')
    plt.ylabel('RT residuals')
    plt.title('LOWESS Fitting Results')
    plt.legend()
    plt.grid()
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


def plt_rt_dev(rt_dev_matrix, save_path):
    plt.rcParams['savefig.dpi'] = 1200
    plt.plot(rt_dev_matrix.T, linewidth=0.1, alpha=0.1)
    plt.savefig(os.path.join(save_path, 'pu_estimation_raw.png'))
    plt.clf()


def plt_pu(normed_rt_dev_matrix, pu, save_path):
    plt.rcParams['savefig.dpi'] = 1200
    plt.plot(normed_rt_dev_matrix.T, linewidth=0.1, alpha=0.1)
    plt.plot(pu, linewidth=3)
    plt.savefig(os.path.join(save_path, 'pu_estimation_normed.png'))
    plt.clf()

