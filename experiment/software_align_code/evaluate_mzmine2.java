package net.sf.mzmine.modules.peaklistmethods.alignment.ransac;

import com.google.common.collect.BoundType;
import com.google.common.collect.Range;
import com.opencsv.CSVWriter;
import net.sf.mzmine.datamodel.*;
import net.sf.mzmine.datamodel.impl.SimpleDataPoint;
import net.sf.mzmine.datamodel.impl.SimpleFeature;
import net.sf.mzmine.datamodel.impl.SimplePeakList;
import net.sf.mzmine.datamodel.impl.SimplePeakListRow;
import net.sf.mzmine.modules.peaklistmethods.alignment.join.JoinAlignerParameters;
import net.sf.mzmine.modules.peaklistmethods.alignment.join.JoinAlignerTask;
import net.sf.mzmine.parameters.ParameterSet;
import net.sf.mzmine.parameters.parametertypes.selectors.PeakListsSelectionType;
import net.sf.mzmine.parameters.parametertypes.tolerances.MZTolerance;
import net.sf.mzmine.parameters.parametertypes.tolerances.RTTolerance;
import net.sf.mzmine.project.impl.MZmineProjectImpl;
import net.sf.mzmine.project.impl.RawDataFileImpl;
import weka.core.pmml.jaxbbindings.False;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class evaluate_mzmine2 {

    public static void eval(String[] csvFilePaths, String outputFilePath, double mz_tolerance, double ppm_mz_tolerance,
                            double rt_tolerance_before, double rt_tolerance_after, double margin, String method) {
        RawDataFile[] rawDataFiles = new RawDataFile[csvFilePaths.length];
        MZmineProject project = new MZmineProjectImpl();
        PeakList[] peakLists = new PeakList[csvFilePaths.length];
        try {
            for (int i = 0; i < csvFilePaths.length; i ++) {
                String[] pathSplit = csvFilePaths[i].split("\\\\");
                String fileName = pathSplit[pathSplit.length - 1].split("\\.")[0];
                RawDataFile rawDataFile = new RawDataFileImpl(fileName);
                rawDataFiles[i] = rawDataFile;
                PeakList peakList = new SimplePeakList(fileName, rawDataFile);
                BufferedReader reader = new BufferedReader(new FileReader(csvFilePaths[i]));
                int rowId = 0;
                String line = null;
                while ((line = reader.readLine()) != null) {
                    String[] lineSplit = line.strip().split(",");
                    PeakListRow row = new SimplePeakListRow(rowId);
                    Feature peak = new SimpleFeature(rawDataFile, Double.parseDouble(lineSplit[0]),
                            Double.parseDouble(lineSplit[1]), Double.parseDouble(lineSplit[2]),
                            Double.parseDouble(lineSplit[2]), null, new DataPoint[]{new SimpleDataPoint(0,0)},
                            Feature.FeatureStatus.DETECTED, 0, 0,
                            null, null, null,
                            Range.range(0.0, BoundType.CLOSED, Double.parseDouble(lineSplit[2]), BoundType.CLOSED));
                    row.addPeak(peak.getDataFile(), peak);
                    peakList.addRow(row);
                    rowId ++;
                }
                peakLists[i] = peakList;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        MZTolerance mzTolerance  = new MZTolerance(mz_tolerance, ppm_mz_tolerance);
        RTTolerance rtToleranceBefore = new RTTolerance(true, rt_tolerance_before);
        RTTolerance rtToleranceAfter = new RTTolerance(true, rt_tolerance_after);
        switch (method) {
            case "ransac":
                ParameterSet ransacParams = new RansacAlignerParameters();
                ransacParams.getParameter(RansacAlignerParameters.MZTolerance).setValue(mzTolerance);
                ransacParams.getParameter(RansacAlignerParameters.RTToleranceBefore).setValue(rtToleranceBefore);
                ransacParams.getParameter(RansacAlignerParameters.RTToleranceAfter).setValue(rtToleranceAfter);

                ransacParams.getParameter(RansacAlignerParameters.peakListName).setValue("RANSAC");
                ransacParams.getParameter(RansacAlignerParameters.SameChargeRequired).setValue(false);

                ransacParams.getParameter(RansacAlignerParameters.Iterations).setValue(100000);
                ransacParams.getParameter(RansacAlignerParameters.NMinPoints).setValue(0.5);
                ransacParams.getParameter(RansacAlignerParameters.Margin).setValue(margin);
                ransacParams.getParameter(RansacAlignerParameters.Linear).setValue(false);

                RansacAlignerTask ransacTask = new RansacAlignerTask(project, peakLists, ransacParams);
                ransacTask.run();
                break;
            case "join":
                ParameterSet joinParams = new JoinAlignerParameters();
                joinParams.getParameter(JoinAlignerParameters.MZTolerance).setValue(mzTolerance);
                joinParams.getParameter(JoinAlignerParameters.MZWeight).setValue(1d);
                joinParams.getParameter(JoinAlignerParameters.RTTolerance).setValue(rtToleranceAfter);
                joinParams.getParameter(JoinAlignerParameters.RTWeight).setValue(1d);

                joinParams.getParameter(JoinAlignerParameters.peakListName).setValue("Join");
                joinParams.getParameter(JoinAlignerParameters.peakLists).setValue(PeakListsSelectionType.SPECIFIC_PEAKLISTS, peakLists);
                joinParams.getParameter(JoinAlignerParameters.SameChargeRequired).setValue(false);
                joinParams.getParameter(JoinAlignerParameters.SameIDRequired).setValue(false);
                JoinAlignerTask joinTask = new JoinAlignerTask(project, joinParams);
                joinTask.run();
                break;
        }



        PeakListRow[] resultList = project.getPeakLists()[0].getRows();
        try {
            CSVWriter csvWriter = new CSVWriter(new FileWriter(outputFilePath));
            String[] firstLine = new String[4 + 3 * csvFilePaths.length];
            firstLine[0] = "mz";
            firstLine[1] = "rt";
            firstLine[2] = "area";
            firstLine[3] = "#";
            for (int i = 0; i < rawDataFiles.length; i ++) {
                firstLine[4 + 3 * i] = rawDataFiles[i].getName() + "_mz";
                firstLine[5 + 3 * i] = rawDataFiles[i].getName() + "_rt";
                firstLine[6 + 3 * i] = rawDataFiles[i].getName() + "_area";
            }
            csvWriter.writeNext(firstLine, false);
            for (PeakListRow peakListRow: resultList) {
                if (peakListRow.getPeaks() == null || peakListRow.getPeaks().length == 0) {
                    continue;
                }
                String[] line = new String[4 + 3 * csvFilePaths.length];
                double mzSum = 0d, rtSum = 0d, areaSum = 0d;
                for (int i = 0; i < rawDataFiles.length; i ++) {
                    Feature peak = peakListRow.getPeak(rawDataFiles[i]);
                    if (peak != null) {
                        mzSum += peak.getMZ();
                        rtSum += peak.getRT();
                        areaSum += peak.getArea();
                        line[4 + 3 * i] = Double.toString(peak.getMZ());
                        line[5 + 3 * i] = Double.toString(peak.getRT());
                        line[6 + 3 * i] = Double.toString(peak.getArea());
                    } else {
                        line[4 + 3 * i] = "0.0";
                        line[5 + 3 * i] = "0.0";
                        line[6 + 3 * i] = "0.0";
                    }
                }
                line[0] = Double.toString(mzSum / peakListRow.getPeaks().length);
                line[1] = Double.toString(rtSum / peakListRow.getPeaks().length);
                line[2] = Double.toString(areaSum / peakListRow.getPeaks().length);
                line[3] = "0";
                csvWriter.writeNext(line, false);
            }
            csvWriter.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("CSV saved");
    }

    public static void main(String[] args) {
        String[] featureSources = {"metapro", "mzmine", "openms", "xcms", "dinosaur", "maxquant"};
        String[] dataNames = {"AT", "EC_H", "Benchmark_FC", "UPS_M", "UPS_Y"};//"MTBLS3038_NEG", "MTBLS3038_POS", "MTBLS5430_Lip_NEG", "MTBLS5430_Lip_POS","MTBLS5430_Metabo_NEG", "MTBLS5430_Metabo_POS"
        double[] mz_tolerances = {0.03, 0.005, 0.02, 0.025, 0.015};//0.05, 0.05, 0.015, 0.03,0.002, 0.03
        double[] rt_befores = {1.0, 1.0, 1.0, 0.6, 1.0};//0.2, 0.2, 0.15, 0.15, 0.1, 0.15
        double[] rt_afters = {3.0, 0.4, 0.4, 1.0, 3.0};//0.25, 0.1, 0.2, 0.4, 0.25, 0.2
        int n = 0;
        for (String dataName : dataNames) {
            for (String featureSource : featureSources) {
                File dataFile = new File("E:\\workspace\\" + dataName + "\\" + featureSource);
                File[] dataFiles = dataFile.listFiles();
                double mz_tolerance = mz_tolerances[n];
                double rt_before = rt_befores[0];
                double rt_after = rt_afters[0];

                if (dataFiles != null) {
                    String[] filePaths = new String[dataFiles.length];
                    for (int i = 0; i < dataFiles.length; i++) {
                        filePaths[i] = dataFiles[i].getPath();
                    }

                    String outputPathRansac = "E:\\workspace\\" + dataName + "_results_" + featureSource + "\\mzmine2\\" + dataName + "_aligned_mzmine2_ransac.csv";
                    long startTime = System.currentTimeMillis();
                    eval(filePaths, outputPathRansac, mz_tolerance, 0.0, rt_before, rt_after, 0.05, "ransac");
                    System.out.println(System.currentTimeMillis() - startTime);

                    String outputPathJoin = "E:\\workspace\\" + dataName + "_results_" + featureSource + "\\mzmine2\\" + dataName + "_aligned_mzmine2_join.csv";
                    startTime = System.currentTimeMillis();
                    eval(filePaths, outputPathJoin, mz_tolerance, 0.0, rt_before, rt_after, 0.05, "join");
                    System.out.println(System.currentTimeMillis() - startTime);
                }
            }
            n += 1;
        }
    }
}

