# Copyright (c) [2025] [Ruimin Wang, Shouyang Ren, Etienne Caron and Changbin Yu]
# Trend-Aligner is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde
from src.fine_feature_matching import _find_matches
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.linalg import lstsq
from src.utils import plot_utils
import warnings

def _lowess(from_data, to_data, params, save_path, plot):
    candidate_from_rts = []
    candidate_to_rts = []
    # rt, mz, area
    for to_data_idx in range(len(to_data)):
        if to_data[to_data_idx][1] < params.from_rt or to_data[to_data_idx][1] > params.to_rt:
            continue
        if params.use_ppm:
            tmp_mz_tolerance = to_data[to_data_idx][0] * params.mz_tolerance * 1e-6
        else:
            tmp_mz_tolerance = params.mz_tolerance
        target = to_data[to_data_idx]
        match_indices = _find_matches(from_data, target[0], target[1], tmp_mz_tolerance, params.rt_tolerance)
        if len(match_indices) == 0:
            continue

        candidate_to_rts.append(to_data[to_data_idx][1])
        if len(match_indices) == 1:
            candidate_from_rts.append(from_data[match_indices[0]][1])
        else:
            match_indices = match_indices[np.argsort(-from_data[match_indices, 2])]
            candidate_from_rts.append(from_data[match_indices[0]][1])

    candidate_from_rts = np.array(candidate_from_rts)
    candidate_to_rts = np.array(candidate_to_rts)

    df = pd.DataFrame({
        'x': candidate_from_rts,
        'y': candidate_to_rts
    }).drop_duplicates().sort_values('x')

    rt_min, rt_max = df['x'].min(), df['x'].max()
    n_blocks = 20
    bins = np.linspace(rt_min, rt_max, n_blocks + 1)
    df['y-x'] = df['y'] - df['x']
    df['block'] = pd.cut(df['x'], bins=bins, labels=False, include_lowest=True)

    list_x = []
    list_y = []
    for i in range(n_blocks):
        block_df = df[df['block'] == i]
        if len(block_df) < 100:
            list_x = list_x + block_df['x'].tolist()
            list_y = list_y + block_df['y'].tolist()
            continue
        block_df = block_df.reset_index(drop=True)

        kde = gaussian_kde(block_df['y-x'], bw_method=0.1)
        density = kde(block_df['y-x'])
        density = density/max(density)
        np.random.seed(501)
        random_numbers = np.random.rand(len(density))
        score = density - random_numbers
        indices = np.argsort(-score)
        indices = indices[:len(indices)//2]
        list_x = list_x + block_df.loc[indices, 'x'].tolist()
        list_y = list_y + block_df.loc[indices, 'y'].tolist()

    candidate_from_rts = np.array(list_x)
    candidate_to_rts = np.array(list_y)

    if params.frac == 'tPRESS':
        tmp_frac = select_f(candidate_from_rts, candidate_to_rts)
    else:
        tmp_frac = params.frac
    result_lowess = lowess(candidate_to_rts.flatten(), candidate_from_rts.flatten(), frac=tmp_frac, it=2)
    while np.any(np.isnan(result_lowess[:, 1])):
        tmp_frac += 0.1
        result_lowess = lowess(candidate_to_rts.flatten(), candidate_from_rts.flatten(), frac=tmp_frac, it=2)
    unique_values, unique_indices = np.unique(result_lowess[:,0], return_index=True)
    result_lowess = result_lowess[unique_indices]
    lowess_function = interp1d(result_lowess[:,0], result_lowess[:,1], kind='quadratic', fill_value="extrapolate")

    if plot == True:
        plot_utils.plt_lowess_fitting_results(candidate_from_rts, candidate_to_rts, result_lowess, lowess_function, save_path)
    return lowess_function, result_lowess



def lowess_align(result_data_list, coarse_registration_params, save_path, plot):
    func_map = {}
    warp_data = {}
    for i in range(len(result_data_list)):
        if i != coarse_registration_params.centric_idx:
            continue
        if i not in func_map.keys():
            func_map[i] = {}
        for j in range(len(result_data_list)):
            if j == i:
                func_map[j] = None
                warp_data[j] = None
                continue
            func, lowess_data = _lowess(result_data_list[j], result_data_list[i], coarse_registration_params, save_path, plot)
            func_map[j] = func
            warp_data[j] = lowess_data

    func_list = []
    warped_data = []
    for i in range(len(result_data_list)):
        ori_rts = result_data_list[i][:, 1]
        sorted_idxes = np.argsort(ori_rts)
        sorted_ori_rts = ori_rts[sorted_idxes]
        sorted_intensities = result_data_list[i][:, 2][sorted_idxes]
        if i == coarse_registration_params.centric_idx:
            func_list.append(None)
            warped_data.append([sorted_ori_rts, sorted_ori_rts, sorted_intensities])
            continue
        func_list.append(func_map[i])
        warped_rts = func_map[i](sorted_ori_rts)
        warped_data.append([sorted_ori_rts, warped_rts, sorted_intensities])

    return func_list, warped_data



def tricube(x):
    return (1 - np.abs(x)**3)**3 * (np.abs(x) <= 1)


def lowess_fit(x, y, xi, f, d=1, exclude_i=None):
    n = len(x)
    r = max(1, int(round(f * n)))
    distances = np.abs(x - xi)
    nearest_indices = np.argsort(distances)[:r]

    if exclude_i is not None:
        nearest_indices = nearest_indices[nearest_indices != exclude_i]
        if len(nearest_indices) == 0:
            return np.nan
    h = distances[np.argsort(distances)[r-1]]
    if h == 0:
        return y[np.where(x == xi)[0][0]]
    weights = tricube((x[nearest_indices] - xi) / h)
    X = np.array([(x[nearest_indices] - xi)**j for j in range(d+1)]).T
    W = np.diag(weights)

    # Weighted Least Squares Fitting
    try:
        beta, _, _, _ = lstsq(W @ X, W @ y[nearest_indices])
    except np.linalg.LinAlgError:
        return np.nan

    return np.sum(beta * (xi - xi)**np.arange(d+1))


def compute_press(x, y, f, d=1, robust_weights=None):
    press = 0.0
    n = len(x)
    for i in range(n):
        y_pred = lowess_fit(x, y, x[i], f, d, exclude_i=i)
        if not np.isnan(y_pred):
            residual = y[i] - y_pred
            if robust_weights is not None:
                press += robust_weights[i] * residual**2
            else:
                press += residual**2
    return press


def select_f(x, y, d=1, max_iter=3, f_bounds=(0.1, 0.9)):
    if len(x) > 200:
        np.random.seed(501)
        indices = np.random.choice(
            len(x),
            size=200,
            replace=True,
        )
        x = x[indices]
        y = y[indices]

    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance.")

    # Step 1: initial non-robust PRESS optimization
    res = minimize_scalar(
        lambda f: compute_press(x, y, f, d=d),
        bounds=f_bounds,
        method='bounded',
        tol=0.05
    )
    f_current = res.x
    robust_weights = np.ones(len(x))

    # Step 2-3: robust iterative optimization
    for _ in range(max_iter):
        residuals = np.array([y[i] - lowess_fit(x, y, x[i], f_current, d) for i in range(len(x))])
        s = np.median(np.abs(residuals))
        u = residuals / (6 * s)
        robust_weights = (1 - u**2)**2 * (np.abs(u) <= 1)

        # updated PRESS optimization (with robust weights)
        res = minimize_scalar(
            lambda f: compute_press(x, y, f, d=d, robust_weights=robust_weights),
            bounds=f_bounds,
            method='bounded',
            tol=0.05
        )
        f_new = res.x
        if np.abs(f_new - f_current) < 0.05:
            break
        f_current = f_new

    return f_current