%%
% Load the data
Datanames = {'AT', 'EC_H', 'Benchmark_FC', 'UPS_M', 'UPS_Y'}; % 'MTBLS3038_NEG', 'MTBLS3038_POS', 'MTBLS5430_Lip_NEG', 'MTBLS5430_Lip_POS', 'MTBLS5430_Metabo_NEG', 'MTBLS5430_Metabo_POS'
featureSources = {'metapro', 'mzmine2', 'openms', 'xcms', 'dinosaur', 'maxquant'}; % 'metapro', 'mzmine2', 'openms', 'xcms', , 'AntDAS'
timeCosts = struct();
FIadjustMethod = 'regression'; % {'none','median','regression'}
RT_befores = {1, 1, 1, 0.6, 1};% 0.2, 0.2, 0.15, 0.15, 0.1, 0.15
RT_afters = {3, 0.4, 0.4, 1, 3};% 0.25, 0.1, 0.2, 0.4, 0.25, 0.2
MZ_befores = {0.03, 0.005, 0.02, 0.015, 0.015};% 0.05, 0.03, 0.015, 0.03, 0.002, 0.03
MZ_afters = {0.03, 0.005, 0.02, 0.025, 0.015};% 0.05, 0.05, 0.015, 0.03, 0.002, 0.03
log10FI_befores = {3, 0.6, 0.2, 2, 3};% 2.5, 2, 3.5, 4, 2, 2
log10FI_afters = {2, 0.6, 0.6, 1.5, 1};% 2.5, 3.5, 1, 1.5, 1, 1.5
n = 1;
for i = 1:length(Datanames)
    Dataname = Datanames{i};  
    RT_before = RT_befores{n};
    RT_after = RT_afters{n};
    MZ_before = MZ_befores{n};
    MZ_after = MZ_afters{n};
    log10FI_before = log10FI_befores{n};
    log10FI_after = log10FI_afters{n};

    for j = 1:length(featureSources)
        featureSource = featureSources{j};
        fprintf('Data name: %s, Feature source: %s\n', Dataname, featureSource);
        file_list = dir(fullfile('E:\workspace', [Dataname, '_results_', featureSource], 'M2S', '*.csv'));
        refFilename = fullfile(file_list(1).folder, file_list(1).name);
        targetFilenames = arrayfun(@(x) fullfile(x.folder, x.name), file_list(2:end), 'UniformOutput', false);
        tic;
        for i = 1:length(targetFilenames)
            fprintf('Data name: %s, Feature source: %s\n', Dataname, featureSource);
            fprintf('這是第 %d 次循環\n', i);
            targetFilename = targetFilenames{i};
            [refFeatures] = importdata(refFilename);
            [targetFeatures] = importdata(targetFilename);
        
            [refMZRT_str] = M2S_createLabelMZRT('ref',refFeatures(:,2),refFeatures(:,1));
            [targetMZRT_str] = M2S_createLabelMZRT('target',targetFeatures(:,2),targetFeatures(:,1));
        
           
            % Procedure part 1: find all possible matches
            % create a structure to keep the options chosen at each step
            opt = struct;
        
            [refSet,targetSet,Xr_connIdx,Xt_connIdx, opt]=M2S_matchAll(refFeatures,targetFeatures);
        
            % M2S_matchAll SETTINGS for the two main examples:
            opt.FIadjustMethod = FIadjustMethod;
            opt.multThresh.RT_intercept = [-RT_before, RT_after];
            opt.multThresh.RT_slope = [0 0];
            opt.multThresh.MZ_intercept = [-MZ_before MZ_after]; % m/z units
            opt.multThresh.MZ_slope = [-5e-6 5e-6]; % ppm
            opt.multThresh.log10FI_intercept = [-log10FI_before log10FI_after];
            opt.multThresh.log10FI_slope = [0 1];
        
            plotType = 0; 
            [refSet,targetSet,Xr_connIdx,Xt_connIdx,opt]=M2S_matchAll(refFeatures,targetFeatures,opt.multThresh,opt.FIadjustMethod,plotType);
        
            % Procedure part 2: Calculate penalisation scores for each match
            % Find neighbours, the RT/MZ/log10FI trends, and the residuals - SETTINGS OPTIONAL
            opt.neighbours.nrNeighbors = 5;
            opt.calculateResiduals.neighMethod = 'circle';
            opt.pctPointsLoess = 0;% no loess
            plotTypeResiduals = 0;
            [Residuals_X,Residuals_trendline] = M2S_calculateResiduals(refSet,targetSet,Xr_connIdx,Xt_connIdx,opt.neighbours.nrNeighbors, opt.calculateResiduals.neighMethod,opt.pctPointsLoess,plotTypeResiduals);
        
            % Adjust the residuals
            opt.adjustResiduals.residPercentile = [0.1,0.01,1.5];
            [adjResiduals_X,residPercentile] = M2S_adjustResiduals(refSet,targetSet,Residuals_X,opt.adjustResiduals.residPercentile);
        
            % Adjust the weight of each dimension (RT, MZ, log10FI), get penalisation scores
            opt.weights.W = [1,1,1]; % equal weight
            [penaltyScores] = M2S_defineW_getScores(refSet,targetSet,adjResiduals_X,opt.weights.W,1);
        
            % Decide the best of the multiple matches
            [eL,eL_INFO,CC_G1]= M2S_decideBestMatches(refSet, targetSet, Xr_connIdx,Xt_connIdx, penaltyScores);
        
        
            % Procedure part 3: find false positives (tighten thresholds)
            opt.falsePos.methodType = 'trend_mad'; %{'none','scores','byBins','trend_mad','residuals_mad'} 
            opt.falsePos.nrMad = 5;
            plotOrNot = 0;
            [eL_final, eL_final_INFO] = M2S_findPoorMatches(eL,refSet,targetSet,opt.falsePos.methodType,opt.falsePos.nrMad,plotOrNot);
            
        
            refFeatures_idx = eL_final.Xr_connIdx(eL_final.notFalsePositives == 1);
            targetFeatures_idx = eL_final.Xt_connIdx(eL_final.notFalsePositives == 1);
        
        
            % The final matched datasets are:
            refMatched = refFeatures(refFeatures_idx,:);
            refMatched_MZRTstr = refMZRT_str(refFeatures_idx);
            targetMatched = targetFeatures(targetFeatures_idx,:);
            targetMatched_MZRTstr = targetMZRT_str(targetFeatures_idx);
        
            refTable = [table(refMatched_MZRTstr),array2table(refMatched,'VariableNames',{'rtmed','mzmed','fimed'})];
            targetTable = [table(targetMatched_MZRTstr),array2table(targetMatched,'VariableNames',{'rtmed','mzmed','fimed'})];
        
            % writetable(refTable,'M2S_datasetsMatched.xlsx','Sheet',4)
            % writetable(targetTable,'M2S_datasetsMatched.xlsx','Sheet',8)
            newSheetName1 = ['Ref_', num2str(i)];
            newSheetName2 = ['Target_', num2str(i)];
            writetable(refTable, fullfile('E:\workspace', [Dataname, '_results_', featureSource], 'M2S\M2S_datasetsMatched.xlsx'), 'Sheet', newSheetName1)
            writetable(targetTable, fullfile('E:\workspace', [Dataname, '_results_', featureSource], 'M2S\M2S_datasetsMatched.xlsx'),'Sheet', newSheetName2)
        
        
        end
        timecost = toc;
        fieldName = sprintf('%s_%s', Dataname, featureSource);
        timeCosts.(fieldName) = timecost; 
    end
    n = n +1;
end
