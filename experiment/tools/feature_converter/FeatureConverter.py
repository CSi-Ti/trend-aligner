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
import pandas as pd
import csv
from pyopenms import FeatureMap, FeatureXMLFile


class FeatureConverter:
    """
    FeatureConverter: Converts feature lists from various software tools to a standardized CSV format
    (m/z, rt, intensity, m/z_start, m/z_end, rt_start, rt_end)

    Supported software: AntDAS(.mzML_msePeakDectection.xlsx), Dinosaur(.features.tsv), MaxQuant(allPeptides.txt),
                    MZmine2(_chromatograms_deconvoluted.csv or custom suffix), OpenMS(.featureXML), XCMS(.csv)

    :param path: Path to the input feature list file
    :param software: Software name ('antdas', 'dinosaur', 'maxquant', 'mzmine2', 'openms', 'xcms')
    :param save_folder: Output directory. If None, saves in the same folder as input file. Defaults to None.
    :param suffix: Remove file suffix from file basename. Use default for the software if None.
    :param separator: File delimiter character. Defaults to '\t'.
    """

    def __init__(self, path, software, save_folder=None, suffix=None, separator=','):
        self.path = path
        self.software = software.lower()
        self.save_folder = save_folder
        self.suffix = suffix
        self.separator = separator

        # Set default suffix if not provided
        if self.suffix is None:
            self.suffix = self._get_default_suffix()

    def _get_default_suffix(self):
        """Get default file suffix for each supported software"""
        suffix_map = {
            'antdas': '.mzML_msePeakDectection.xlsx',
            'dinosaur': '.features.tsv',
            'maxquant': '.txt',  # MaxQuant uses allPeptides.txt
            'mzmine2': '.mzML_chromatograms_deconvoluted.csv',
            'openms': '.featureXML',
            'xcms': '.csv'
        }
        return suffix_map.get(self.software, '')

    def convert(self):
        """Route to the appropriate converter based on software type"""
        converter_map = {
            'antdas': self._convert_antdas,
            'dinosaur': self._convert_dinosaur,
            'maxquant': self._convert_maxquant,
            'mzmine2': self._convert_mzmine2,
            'openms': self._convert_openms,
            'xcms': self._convert_xcms
        }

        if self.software not in converter_map:
            raise ValueError(
                f"Unsupported software: {self.software}. Supported options are: {', '.join(converter_map.keys())}")

        try:
            converter_map[self.software]()
        except FileNotFoundError:
            print(f"Error: The file {self.path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _get_output_path(self):
        """Get output file path and sample name"""
        sample_name = os.path.basename(self.path).split(self.suffix)[0]
        if self.save_folder:
            out_path = os.path.join(self.save_folder, sample_name + '.csv')
        else:
            out_path = os.path.join(os.path.dirname(self.path), sample_name + '.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return out_path, sample_name

    def _convert_antdas(self):
        """Convert AntDAS feature list"""
        df = pd.read_excel(self.path)
        df = df[['m/z', 'RT', 'Area']]
        df['m/z_start'] = df['m/z'] - 0.001
        df['m/z_end'] = df['m/z'] + 0.001
        df['RT_start'] = df['RT'] - 0.05
        df['RT_end'] = df['RT'] + 0.05
        df = df[df['Area'] != 0]

        out_path, sample_name = self._get_output_path()
        df.to_csv(out_path, header=False, index=False)
        print(f"{sample_name} conversion finished.")

    def _convert_dinosaur(self):
        """Convert Dinosaur feature list"""
        df = pd.read_csv(self.path, sep=self.separator)
        df['mz_min'] = df['mz'] - 0.001
        df['mz_max'] = df['mz'] + 0.001
        df = df[['mz', 'rtApex', 'intensitySum', 'mz_min', 'mz_max', 'rtStart', 'rtEnd']]
        out_path, sample_name = self._get_output_path()
        df.to_csv(out_path, header=False, index=False)
        print(f"{sample_name} conversion finished.")

    def _convert_maxquant(self):
        """Convert MaxQuant feature list"""
        df = pd.read_csv(self.path, sep=self.separator)
        grouped_data = df.groupby(df.columns[0])

        for group_name, group_df in grouped_data:
            out_path = os.path.join(self.save_folder if self.save_folder else os.path.dirname(self.path),
                                    group_name + '.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            group_df["m/z_start"] = group_df["m/z"] - 0.001
            group_df["m/z_end"] = group_df["m/z"] + 0.001
            group_df["Retention_time_start"] = group_df["Retention time"] - 0.05
            group_df["Retention_time_end"] = group_df["Retention time"] + 0.05
            group_df = group_df[["m/z", "Retention time", "Intensity",
                                 "m/z_start", "m/z_end",
                                 "Retention_time_start", "Retention_time_end"]]

            group_df.to_csv(out_path, header=False, index=False)
            print(f"{group_name} conversion finished.")

    def _convert_mzmine2(self):
        """Convert MZmine2 feature list"""
        df = pd.read_csv(self.path, sep=self.separator)
        df = df[['row m/z', 'row retention time', '* Peak area', '* Peak m/z min', '* Peak m/z max', '* Peak RT start', '* Peak RT end']]
        out_path, sample_name = self._get_output_path()
        df.to_csv(out_path, header=False, index=False)
        print(f"{sample_name} conversion finished.")

    def _convert_openms(self):
        """Convert OpenMS feature list"""
        featureMap = FeatureMap()
        FeatureXMLFile().load(self.path, featureMap)

        out_path, sample_name = self._get_output_path()
        with open(out_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='unix', quoting=csv.QUOTE_NONE, quotechar='')
            for feature in featureMap:
                writer.writerow([
                    feature.getMZ(),
                    feature.getRT() / 60,
                    feature.getIntensity(),
                    feature.getMZ() - 0.001,
                    feature.getMZ() + 0.001,
                    feature.getRT() / 60 - 0.05,
                    feature.getRT() / 60 + 0.05
                ])
        print(f"{sample_name} conversion finished.")

    def _convert_xcms(self):
        """Convert XCMS feature list"""
        df = pd.read_csv(self.path, sep=self.separator)
        df = df[["mz", "rt", "into", "mzmin", "mzmax", "rtmin", "rtmax"]]
        df[["rt", "rtmin", "rtmax"]] = df[["rt", "rtmin", "rtmax"]] / 60
        out_path, sample_name = self._get_output_path()
        df.to_csv(out_path, header=False, index=False)
        print(f"{sample_name} conversion finished.")

if __name__ == '__main__':
    # AntDAS
    converter = FeatureConverter(path="sample.mzML_msePeakDectection.xlsx", software="antdas")
    converter.convert()

    # Dinosaur
    converter = FeatureConverter(path="sample.features.tsv", software="dinosaur", separator='\t')
    converter.convert()

    # MaxQuant
    converter = FeatureConverter(path="allPeptides.txt", software="maxquant", separator='\t')
    converter.convert()

    # MZmine2
    converter = FeatureConverter(path="sample.mzML_chromatograms_deconvoluted.csv", software="mzmine2")
    converter.convert()

    # OpenMS
    converter = FeatureConverter(path="sample.featureXML", software="openms")
    converter.convert()

    # XCMS
    converter = FeatureConverter(path="sample.csv", software="xcms", separator=',')
    converter.convert()