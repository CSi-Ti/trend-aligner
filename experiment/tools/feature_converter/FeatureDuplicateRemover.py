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

class FeatureDuplicateRemover:
    """
    FeatureDuplicateRemover: Removes features in standardized CSV format files where mz_diff < 0.001 and rt_diff < 0.05

    :param path: Path to the standardized CSV format file. The output file will overwrite the original file.
    """

    def __init__(self, path):
        self.path = path

    def remover(self):
        try:
            print(f"Processing file: {self.path}")
            df = pd.read_csv(self.path, header=None, separator=',')
            df = df.sort_values(by=0).reset_index(drop=True)
            df['mz_diff'] = df[0].diff().abs()
            groups_col1 = (df['mz_diff'] >= 0.001).cumsum()
            sorted_df = df.groupby(groups_col1)
            sorted_df = sorted_df.apply(lambda x: x.sort_values(by=1)).reset_index(drop=True)
            sorted_df['mz_diff'] = sorted_df[0].diff().abs()
            sorted_df['rt_diff'] = sorted_df[1].diff().abs()

            condition = (sorted_df['mz_diff'] < 0.001) & (sorted_df['rt_diff'] < 0.05)
            indices = sorted_df.index[condition]
            if not indices.empty:
                new_indices = indices - 1
                combined_indices = indices.append(pd.Index(new_indices)).unique()
                indices_df = pd.DataFrame(combined_indices, columns=['index'])
                indices_df = indices_df.sort_values(by="index").reset_index(drop=True)
                indices_df['group'] = (indices_df['index'].diff() != 1).cumsum()
                grouped_indices = indices_df.groupby('group')['index'].apply(list).reset_index(drop=True)
                final_indices = []
                for group in grouped_indices:
                    indices = group
                    values = sorted_df.iloc[indices, 2]
                    max_index = values.idxmax()
                    for index in indices:
                        if index != max_index:
                            final_indices.append(index)
                            print(f"Number of duplicates found: {len(final_indices)}")
            else:
                final_indices = []
                print("No duplicates found.")

            result_df = sorted_df.drop(index=final_indices)
            result_df = result_df.drop(columns=['mz_diff', 'rt_diff'])
            result_df.to_csv(self.path, header=False, index=False)
            print(f"Processed file saved: {self.path}")
        except FileNotFoundError:
            print(f"Error: The file {self.path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    remover = FeatureDuplicateRemover(path="sample.csv")
    remover.remover()