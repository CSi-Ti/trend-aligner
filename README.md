# Trend-Aligner: a retention time modeling-based feature alignment method for untargeted LC-MS-based data analysis


## Highlights
- **Novelty:** Trend-Aligner models the RT shift as global RT shift and local RT shift. The global RT shift represents the systematic chromatographic condition changes across samples, while the local RT shift represents the differential response of compounds with different physicochemical properties to chromatographic condition variations. The global RT shift is minimized through locally-weighted scatterplot smoothing regression, and the local RT shift is modeled on latent factor model. 
- **Accuracy:** In the reference-based accuracy benchmarking, Trend-Aligner demonstrated superior performance by outperforming 11 widely-used alignment algorithms from prominent platforms including OpenMS, MZmine2, XCMS, AntDAS, DeepRTAlign, and M2S, achieving the best alignment accuracy, precision and recall across eight metabolomic and five proteomic datasets.
- **Reliability:** In the application-oriented utility validation, Trend-Aligner achieved a 13.9% increase in the number of identified peptides compared to MaxQuant while maintaining stable and precise quantification accuracy. 
- **Efficiency:** Although Trend-Aligner requires iterative optimization for LOWESS bandwidth self-adaptation and involves complex parameter computation in latent factor model via beam search and Gaussian kernel density estimation, Trend-Aligner's processing speed still ranked among the top tier of all currently popular algorithms. The average runtime of Trend-Aligner was around 30 seconds.
- **Open source:** We open-sourced Trend-Aligner under a permissive license to promote the accuracy of MS data analysis more broadly.
- **Dataset:** We manually annotated the feature datasets for eight metabolomic and five proteomic datasets, including 4419 concensus feature and 85084 feature peak with associated m/z, RT, and peak area information. These comprehensively annotated datasets serve as valuable benchmarks for evaluating feature detection, quantification, and alignment accuracy.

## Trend-Aligner workflow
![Image](https://github.com/user-attachments/assets/2cf19cc6-518f-4425-ae32-8babb55a2f69)


## Datasets
The data distributions of evaluation datasets, feature lists, alignment results, manual annotated results, RAB results, runtime, AUV results and parameters settings are available in [Zenodo](https://doi.org/10.5281/zenodo.15054538).


## Setup
1. Prepare the python environment based on your system and hardware.
   
2. Install the dependencies. Here we use ROOT_PATH to represent the root path of Trend-Aligner.
   
    ```cd ROOT_PATH```
   
    ```pip install -r requirements.txt```



## Run Trend-Aligner

### Supported formats
Trend-Aligner processes feature lists comprising m/z and RT information as its primary input. The algorithm accommodates both user-specified delimited formats (CSV/TXT) and native output formats from widely-used LC-MS data analysis platforms, including OpenMS, MZmine2, Dinosaur, XCMS, MaxQuant, and AntDAS-DDA, ensuring broad compatibility with existing data processing pipelines.
### FeatureListReadingParams
```
feature_list_folder_path: Path to directory containing feature list files (CSV/TXT or platform-native formats) [required]

skip_line:      Number of header lines to skip when parsing files (0=no header) [default=0]

mz_col_num:     Column index (1-based) containing m/z values [required, typically=1]

rt_col_num:     Column index (1-based) containing RT values [required, typically=2]

area_col_num:   Column index (1-based) containing peak area/intensity values [required, typically=3]
                • Area values are only used for downstream analysis, not for feature alignment
                • The core algorithm requires only m/z and RT information for alignment
```
### CoarseAlignmentParams
```
mz_tolerance:   The m/z tolerance (Da/ppm) in anchor-based pairwise matching [default=0.01]

use_ppm:        Use ppm instead of Da for m/z tolerances [default=False]

centric_idx:    Index of the reference sample used as alignment anchor (0=first sample by ASCII order) [default=0]

rt_tolerance:   The RT tolerance (in minutes) in anchor-based pairwise matching [default=3]

frac:           LOWESS smoothing parameter: float value (0<frac<=1) for manual bandwidth or 'tPRESS' for automatic adaptation [default='tPRESS']
                • The larger frac, the smoother the fitted LOWESS regression curve.
```
### FineMatchingParams
```
beam_mz_tol:    The m/z tolerance (Da/ppm) for adjacent-run feature matching during sample shift coefficient estimation [required, typical range: 0.005-0.03 Da (or 5-20 ppm)]
                • Relatively narrow tolerances within reasonable ranges may lead to an increased proportion of reliable matching groups

beam_rt_tol:    The RT tolerance (in minutes) for adjacent-run feature matching during sample shift coefficient estimation [required]
                • Should be set according to the RT deviations after coarse alignment
                • Relatively narrow tolerances within reasonable ranges may lead to an increased proportion of reliable matching groups
                • Can be estimated via the RT deviation pattern in LOWESS fitting curve plots

match_mz_tol:   The m/z tolerance (Da/ppm) for cross-run feature matching during analyte easy-to-shift coefficient estimation [required, typical range: 0.005-0.03 Da (or 5-20 ppm)]
                • Relatively wider tolerances within reasonable ranges may help prevent missed matches
                • Typically equals or moderately exceeds beam_mz_tol

match_rt_tol:   The RT tolerance (in minutes) for cross-run feature matching during analyte easy-to-shift coefficient estimation [required]
                • Should be set according to the RT deviations after coarse alignment
                • Relatively wider tolerances within reasonable ranges may help prevent missed matches
                • Typically equals or moderately exceeds beam_rt_tol
                • Can be estimated via the RT deviation pattern in LOWESS fitting curve plots

max_rt_tol:     The maximum RT deviation (in minutes) [required]
                • This parameter usually demonstrates good robustness
                • Intentionally larger than match_rt_tol to accommodate RT drift variability
                • Can be estimated via the RT deviation pattern in LOWESS fitting curve plots

use_ppm:        Use ppm instead of Da for m/z tolerances [default=False]
```

### Demos
The package provides the following demonstration datasets and alignment examples:

1. MTBLS733 (QE-HF) Dataset

   Feature extraction platform: MetaPro

2. EC-H (OpenMS) Dataset

   Feature extraction platform: OpenMS

```
Trend-Aligner-master
├── demo
│   ├── metapro_example
│   ├── metapro_result
│   ├── openms_example
│   ├── openms_example_converted
│   ├── openms_result
│   ├── align_demo.py
```

- To run the demo:

```cd ROOT_PATH```

```python demo/align_demo.py```

Feature alignment results are saved in ```result``` and ```openms_result``` folder.


## Citation

Cite our paper at:
```
```

## License

Trend-Aligner is an open-source tool, using [***Mulan Permissive Software License，Version 2 (Mulan PSL v2)***](http://license.coscl.org.cn/MulanPSL2)

## Contacts
For any questions involving Trend-Aligner, please contact us by email.

Ruimin Wang, ruimin.wang@yale.edu

Shouyang Ren, ren_shouyang@163.com

Changbin Yu, yu_lab@sdfmu.edu.cn
