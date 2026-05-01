%% =====================================================================
%  DIABETIC RETINOPATHY BINARY CLASSIFICATION
%  APTOS 2019 Gaussian-Filtered Retinal Fundus Images
%  ---------------------------------------------------------------------
%  Module  : CMP-L014-0 Applications of Data Science
%  Author  : Ravi Shankar Kota
%  Dataset : 3,662 images, 224x224 px, 5 severity classes
%  Task    : Binary classification, No_DR (class 0) vs DR (classes 1-4)
%
%  Pipeline
%    Step 1. Exploratory data analysis on the 5-class dataset
%    Step 2. Binary label conversion (DR vs No_DR)
%    Step 3. Stratified 80/20 split, preprocessing and 5x augmentation
%    Step 4. Feature engineering: HOG (576) + LBP (256) + Gabor (16),
%            Z-score normalisation, PCA retaining 95% variance
%    Step 5. Descriptive and inferential statistics on PCA features
%    Step 6. Multi-information 3D visualisation
%            (3D Feature Space of 5 DR Classes:
%             Position = Mean | Size = Variance | Colour = Class |
%             Error Bars = +/-1 Std | Reference Planes per class)
%    Step 7. Classification: KNN, Linear SVM, Decision Tree,
%            Gaussian Naive Bayes, Random Forest
%
%  Mathematical formulas used in this script
%  ---------------------------------------------------------------------
%   Brightness      mu = (1/N) * sum(I_i)
%   Contrast        sigma = sqrt((1/N) * sum((I_i - mu)^2))
%   Sharpness       sharpness = var( L(I) )
%                   where L = 3x3 Laplacian kernel
%   Skewness        Sk = (1/N) * sum((x - mu)^3) / sigma^3
%   Kurtosis        Kt = (1/N) * sum((x - mu)^4) / sigma^4
%   Grayscale       I_gray = 0.2989*R + 0.5870*G + 0.1140*B
%   Bilinear resize f(x,y) = (1-dy)*[(1-dx)*Q11 + dx*Q21]
%                         + dy*[(1-dx)*Q12 + dx*Q22]
%   Box-Muller      Z = sqrt(-2 ln U1) * cos(2 pi U2)
%   HOG gradient    Gx = I(r,c+1) - I(r,c-1)
%                   Gy = I(r+1,c) - I(r-1,c)
%                   G  = sqrt(Gx^2 + Gy^2)
%                   theta = atan2(Gy, Gx) mod 180
%   HOG L2-norm     h_norm = h / (||h|| + eps)
%   LBP code        LBP(r,c) = sum_{k=0..7} s(n_k - p) * 2^k
%                   where s(x)=1 if x>=0 else 0
%   Gabor filter    g(x',y') = exp(-(x'^2 + gamma^2 y'^2)
%                            m      / (2 sigma^2))
%                            * cos(2 pi f x')
%   Z-score         z = (x - mu_train) / sigma_train
%   Covariance      C = (X' * X) / (n - 1)
%   PCA projection  X_PCA = X * V_k
%   Welch t-test    t = (mu1 - mu2) / sqrt(s1^2/n1 + s2^2/n2)
%   Mann-Whitney    U = R1 - n1*(n1+1)/2
%                   z = (U - mu_U) / sigma_U
%   Conf. interval  CI = mu +/- 1.96 * sigma / sqrt(n)
%   Pearson corr.   r = cov(X,Y) / (sigma_X * sigma_Y)
%   Euclidean dist  d(x,y) = sqrt( sum((x_i - y_i)^2) )
%   SVM hinge loss  L = (1/n) sum max(0, 1 - y(w'x + b))
%                      + (1/2) ||w||^2
%   Gini impurity   G = 1 - p0^2 - p1^2
%   Gaussian NB     P(x|c) = (1/sqrt(2 pi sigma^2))
%                           * exp(-(x - mu)^2 / (2 sigma^2))
%   Accuracy        Acc = (TP + TN) / N
%   Sensitivity     Sens = TP / (TP + FN)
%   Specificity     Spec = TN / (TN + FP)
%   F1-score        F1 = 2*Prec*Sens / (Prec + Sens)
%   AUC-ROC         trapezoidal integral under the ROC curve
% ======================================================================

clear; clc; close all;

%% =====================================================================
%  STEP 1 - EXPLORATORY DATA ANALYSIS (5-CLASS)
% ======================================================================
% Inspect severity distribution and compute three low-level image
% descriptors (brightness, contrast, sharpness) before the binary task.

% 1.1 Dataset path with fallback auto-detection by folder name
datasetPath  = "/MATLAB Drive/Code and Data File/Image Data File/Data File";
targetFolder = "Data File";

if isfolder(datasetPath)
    fprintf('Using user-defined dataset path:\n%s\n', datasetPath);
else
    fprintf('Path not found. Searching for dataset folder automatically...\n');
    currentFolder = pwd;
    folderInfo    = dir(currentFolder);
    found         = false;
    for k = 1:length(folderInfo)
        if folderInfo(k).isdir && strcmp(folderInfo(k).name, targetFolder)
            datasetPath = fullfile(currentFolder, folderInfo(k).name);
            fprintf('Detected dataset folder:\n%s\n', datasetPath);
            found = true;
            break;
        end
    end
    if ~found
        error('Dataset folder "%s" was not found in the current directory.', targetFolder);
    end
end

% 1.2 Load images via imageDatastore; folder names become class labels
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource',       'foldernames');
fprintf('\nDataset loaded successfully.\n');
fprintf('Total Images: %d\n', numel(imds.Files));

% 1.3 Per-class image counts
labelTable = countEachLabel(imds);
disp('--- 5-Class Distribution ---');
disp(labelTable);

% 1.4 Class distribution bar chart
figure('Position', [100 100 700 450]);
bar(labelTable.Count, 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTick', 1:height(labelTable), ...
         'XTickLabel', cellstr(labelTable.Label), 'FontSize', 11);
xlabel('Diabetic Retinopathy Severity Class');
ylabel('Number of Images');
title('5-Class Distribution - APTOS 2019 Dataset');
grid on;

% 1.5 Representative sample images per class
labels     = unique(imds.Labels);
numClasses = numel(labels);
figure('Position', [100 100 800 900]);
plotIndex  = 1;
for i = 1:numClasses
    classIdx = find(imds.Labels == labels(i));
    randIdx  = classIdx(randperm(numel(classIdx), min(2, numel(classIdx))));
    for j = 1:numel(randIdx)
        img = readimage(imds, randIdx(j));
        subplot(numClasses, 2, plotIndex);
        imshow(img);
        title(char(labels(i)), 'Interpreter', 'none', 'FontSize', 9);
        plotIndex = plotIndex + 1;
    end
end
sgtitle('Sample Retinal Fundus Images - Two Per Severity Class');

% 1.6 Image-level descriptors per image
%     brightness = mean intensity
%     contrast   = standard deviation
%     sharpness  = variance of Laplacian
numImages        = numel(imds.Files);
brightnessValues = zeros(numImages, 1);
contrastValues   = zeros(numImages, 1);
sharpnessValues  = zeros(numImages, 1);

% 3x3 discrete Laplacian operator
laplacianFilter  = [0 -1 0; -1 4 -1; 0 -1 0];

for i = 1:numImages
    img = readimage(imds, i);
    if size(img, 3) == 3
        % Grayscale conversion: I_gray = 0.2989*R + 0.5870*G + 0.1140*B
        img     = double(img);
        grayImg = 0.2989*img(:,:,1) + 0.5870*img(:,:,2) + 0.1140*img(:,:,3);
    else
        grayImg = double(img);
    end

    N = numel(grayImg);

    % Brightness: mu = (1/N) * sum(I_i)
    brightnessValues(i) = sum(grayImg(:)) / N;

    % Contrast: sigma = sqrt((1/N) * sum((I_i - mu)^2))
    dev_b = grayImg(:) - brightnessValues(i);
    contrastValues(i) = sqrt(sum(dev_b.^2) / N);

    % Sharpness: sharpness = var( L(I) )
    lapImg = conv2(grayImg, laplacianFilter, 'same');
    mn_lap = sum(lapImg(:)) / numel(lapImg);
    dev_l  = lapImg(:) - mn_lap;
    sharpnessValues(i) = sum(dev_l.^2) / numel(dev_l);
end

% 1.7 Descriptive statistics for 5 classes on 3 descriptors
%     Skewness and kurtosis computed per class
featureData  = {brightnessValues, contrastValues, sharpnessValues};
featureNames = {'Brightness (Mean Intensity)', ...
                'Contrast (Pixel Std Dev)', ...
                'Sharpness (Laplacian Variance)'};

fprintf('\n=== DESCRIPTIVE STATISTICS - 5 CLASSES x 3 FEATURES ===\n');
for fi = 1:3
    fv = featureData{fi};
    fprintf('\n--- Feature: %s ---\n', featureNames{fi});
    fprintf('%-18s %10s %10s %10s %10s %10s\n', ...
        'Class','Mean','Median','Std','Skewness','Kurtosis');
    fprintf('%s\n', repmat('-',1,72));
    for c = 1:numClasses
        idx   = find(imds.Labels == labels(c));
        cVals = fv(idx);
        n     = numel(cVals);
        mu    = sum(cVals)/n;
        xs    = sort(cVals);
        med   = xs(round(0.5*n));
        dev   = cVals - mu;
        sd    = sqrt(sum(dev.^2)/(n-1));
        sk    = (sum(dev.^3)/n)/sd^3;
        kt    = (sum(dev.^4)/n)/sd^4;
        fprintf('%-18s %10.4f %10.4f %10.4f %10.4f %10.4f\n', ...
            char(labels(c)),mu,med,sd,sk,kt);
    end
end

% 1.8 3D scatter of all images coloured by class, with centroids
classColors3D = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.7 0.3; 0.9 0.5 0.1; 0.6 0.2 0.8];
figure('Position', [100 100 900 700]);
hold on;
for c = 1:numClasses
    idx = find(imds.Labels == labels(c));
    col = classColors3D(c,:);
    scatter3(brightnessValues(idx), contrastValues(idx), sharpnessValues(idx), ...
        20, col, 'filled', 'MarkerFaceAlpha', 0.45, 'DisplayName', char(labels(c)));
    % Class centroid at per-class mean
    n_i = numel(idx);
    mB = sum(brightnessValues(idx))/n_i;
    mC = sum(contrastValues(idx))/n_i;
    mS = sum(sharpnessValues(idx))/n_i;
    scatter3(mB, mC, mS, 180, col, 'p', 'filled', ...
        'MarkerEdgeColor','k','LineWidth',1.2,'HandleVisibility','off');
end
xlabel('Brightness'); ylabel('Contrast'); zlabel('Sharpness');
title('3D Feature Scatter - Brightness vs Contrast vs Sharpness (5 Classes)');
legend('Location','best'); grid on; view(35,25); hold off;

% 1.9 Overlaid density histograms for all 5 classes
figure('Position', [80 80 1400 460], 'Color', 'white');
for fi = 1:3
    fv   = featureData{fi};
    gMin = min(fv);  gMax = max(fv);
    nB   = 25;
    edges = linspace(gMin, gMax, nB+1);
    bw    = edges(2) - edges(1);
    ctrs  = (edges(1:nB) + edges(2:end))/2;

    subplot(1, 3, fi); hold on;
    for c = 1:numClasses
        idx = find(imds.Labels == labels(c));
        cv  = fv(idx);
        cnt = zeros(1, nB);
        for v = 1:numel(cv)
            for b = 1:nB
                if cv(v) >= edges(b) && cv(v) < edges(b+1)
                    cnt(b) = cnt(b) + 1; break;
                end
            end
        end
        dens = cnt / (numel(cv) * bw + 1e-9);
        fill([ctrs, fliplr(ctrs)], [dens, zeros(1,nB)], ...
            classColors3D(c,:), 'FaceAlpha', 0.14, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        plot(ctrs, dens, '-o', 'Color', classColors3D(c,:), ...
            'LineWidth', 1.8, 'MarkerSize', 4, 'DisplayName', char(labels(c)));
    end
    xlabel(featureNames{fi}, 'FontSize', 11);
    ylabel('Density', 'FontSize', 11);
    title(featureNames{fi}, 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9); grid on; hold off;
end
sgtitle('Figure 1: Feature Histograms for All 5 Severity Classes Overlaid', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% =====================================================================
%  STEP 2 - BINARY CONVERSION: DR vs No_DR
% ======================================================================
originalLabels = imds.Labels;
binaryLabels   = strings(numel(originalLabels), 1);
for i = 1:numel(originalLabels)
    if strcmp(char(originalLabels(i)), 'No_DR')
        binaryLabels(i) = "No_DR";
    else
        binaryLabels(i) = "DR";
    end
end
imds.Labels = categorical(binaryLabels);

binaryTable = countEachLabel(imds);
disp('--- Binary Class Distribution ---');
disp(binaryTable);

noDrIdx = find(imds.Labels == "No_DR");
drIdx   = find(imds.Labels == "DR");

save('binary_imds.mat','imds');
save('binary_basic_features.mat','brightnessValues','contrastValues','sharpnessValues');
fprintf('\nStep 2 - Binary Conversion Complete.\n');


%% =====================================================================
%  STEP 3 - TRAIN / TEST SPLIT + PREPROCESSING + AUGMENTATION
% ======================================================================
load('binary_imds.mat','imds');
allFiles   = imds.Files;
allLabels  = imds.Labels;
classNames = categories(allLabels);

% 3.1 Stratified 80/20 split
rng(42);
trainFiles={};  trainLabels={};
testFiles={};   testLabels={};
for c = 1:numel(classNames)
    idx    = find(allLabels == classNames{c});
    n      = numel(idx);
    idx    = idx(randperm(n));
    nTrain = round(0.8*n);
    trainFiles  = [trainFiles;  allFiles(idx(1:nTrain))];
    trainLabels = [trainLabels; repmat(classNames(c), nTrain, 1)];
    testFiles   = [testFiles;   allFiles(idx(nTrain+1:end))];
    testLabels  = [testLabels;  repmat(classNames(c), n-nTrain, 1)];
end
fprintf('Raw train: %d  |  Raw test: %d\n', numel(trainFiles), numel(testFiles));

% 3.2 Majority-class undersampling for a balanced training set
trainLabelsCat = categorical(trainLabels);
noDR_idx = find(trainLabelsCat == 'No_DR');
DR_idx   = find(trainLabelsCat == 'DR');
minCount = min(numel(noDR_idx), numel(DR_idx));
noDR_sel = noDR_idx(randperm(numel(noDR_idx), minCount));
DR_sel   = DR_idx(randperm(numel(DR_idx), minCount));
balIdx   = shuffleIdx([noDR_sel; DR_sel]);
trainFiles  = trainFiles(balIdx);
trainLabels = trainLabels(balIdx);
fprintf('Balanced training: %d per class (%d total)\n', minCount, numel(trainFiles));

% 3.3 Preprocess and augment training images (5 versions each)
numTrain   = numel(trainFiles);
augFactor  = 5;
totalAug   = numTrain * augFactor;
trainProcessed = zeros(64, 64, totalAug);
trainLabelsAug = cell(totalAug, 1);

fprintf('Preprocessing and augmenting training images...\n');
for i = 1:numTrain
    img   = imread(trainFiles{i});
    proc  = manualPreprocess(img);
    label = trainLabels{i};
    b     = (i-1)*augFactor + 1;

    trainProcessed(:,:,b)   = proc;                         trainLabelsAug{b}   = label;
    trainProcessed(:,:,b+1) = flipHorizontal(proc);         trainLabelsAug{b+1} = label;
    trainProcessed(:,:,b+2) = flipVertical(proc);           trainLabelsAug{b+2} = label;
    trainProcessed(:,:,b+3) = rotateManual(proc, 15);       trainLabelsAug{b+3} = label;
    trainProcessed(:,:,b+4) = addGaussianNoise(proc, 0.02); trainLabelsAug{b+4} = label;

    if mod(i,500)==0
        fprintf('  Augmented %d / %d\n', i, numTrain);
    end
end
trainLabels = trainLabelsAug;
fprintf('Augmented training set: %d images\n', totalAug);

% 3.4 Preprocess test images (no augmentation)
numTest       = numel(testFiles);
testProcessed = zeros(64, 64, numTest);
fprintf('Preprocessing test images...\n');
for i = 1:numTest
    img = imread(testFiles{i});
    testProcessed(:,:,i) = manualPreprocess(img);
    if mod(i,200)==0
        fprintf('  Processed %d / %d\n', i, numTest);
    end
end
fprintf('Test set: %d images\n', numTest);

% 3.5 Display 5 augmented versions of one training image
figure('Position',[100 100 1000 280]);
sgtitle('Data Augmentation - 5 Versions of a Single Training Image');
augTitles = {'Original','Horizontal Flip','Vertical Flip', ...
             'Rotate +15 deg','Gaussian Noise'};
for v = 1:5
    subplot(1,5,v); imshow(trainProcessed(:,:,v),[]); title(augTitles{v},'FontSize',9);
end

save('preprocessed_data.mat', ...
    'trainProcessed','testProcessed','trainLabels','testLabels', ...
    'trainFiles','testFiles');
fprintf('\nStep 3 - Preprocessing and Augmentation Complete.\n');


%% =====================================================================
%  STEP 4 - FEATURE ENGINEERING
% ======================================================================
load('preprocessed_data.mat');
numTrain = size(trainProcessed, 3);
numTest  = size(testProcessed,  3);
fprintf('\n=== STEP 4: FEATURE ENGINEERING ===\n');

% 4.1 HOG features (576)
fprintf('\nComputing HOG features...\n');
hogTrain = zeros(numTrain, 576);
for i = 1:numTrain
    hogTrain(i,:) = computeHOG(trainProcessed(:,:,i), 8, 9);
    if mod(i,1000)==0, fprintf('  HOG train %d/%d\n',i,numTrain); end
end
hogTest = zeros(numTest, 576);
for i = 1:numTest
    hogTest(i,:) = computeHOG(testProcessed(:,:,i), 8, 9);
end

% 4.2 LBP features (256)
fprintf('Computing LBP features...\n');
lbpTrain = zeros(numTrain, 256);
for i = 1:numTrain
    lbpTrain(i,:) = computeLBP(trainProcessed(:,:,i));
    if mod(i,1000)==0, fprintf('  LBP train %d/%d\n',i,numTrain); end
end
lbpTest = zeros(numTest, 256);
for i = 1:numTest
    lbpTest(i,:) = computeLBP(testProcessed(:,:,i));
end

% 4.3 Gabor features (16)
fprintf('Computing Gabor features...\n');
gaborTrain = zeros(numTrain, 16);
for i = 1:numTrain
    gaborTrain(i,:) = computeGabor(trainProcessed(:,:,i));
    if mod(i,1000)==0, fprintf('  Gabor train %d/%d\n',i,numTrain); end
end
gaborTest = zeros(numTest, 16);
for i = 1:numTest
    gaborTest(i,:) = computeGabor(testProcessed(:,:,i));
end

% 4.4 Concatenation: 576 + 256 + 16 = 848 features per image
XTrain = [hogTrain, lbpTrain, gaborTrain];
XTest  = [hogTest,  lbpTest,  gaborTest];
fprintf('Combined descriptor: %d features per image\n', size(XTrain,2));

% 4.5 Z-score normalisation using training statistics only
featMean = zeros(1, size(XTrain,2));
featStd  = zeros(1, size(XTrain,2));
for j = 1:size(XTrain,2)
    featMean(j) = sum(XTrain(:,j)) / size(XTrain,1);
    dev_j       = XTrain(:,j) - featMean(j);
    featStd(j)  = sqrt(sum(dev_j.^2) / size(XTrain,1));
    if featStd(j) < 1e-8, featStd(j) = 1; end
end
XTrainNorm = (XTrain - repmat(featMean,size(XTrain,1),1)) ./ ...
              repmat(featStd, size(XTrain,1),1);
XTestNorm  = (XTest  - repmat(featMean,size(XTest,1),1))  ./ ...
              repmat(featStd, size(XTest,1),1);

% 4.6 PCA with 95% variance retention
fprintf('\nRunning PCA...\n');
n = size(XTrainNorm, 1);
C = (XTrainNorm' * XTrainNorm) / (n-1);
[V, D]             = eig(C);
eigVals            = diag(D);
[eigVals, sortIdx] = sort(eigVals, 'descend');
V                  = V(:, sortIdx);
cumVar             = cumsum(eigVals) / sum(eigVals);
k95                = find(cumVar >= 0.95, 1);
fprintf('PCA: %d components retain 95%% variance (from %d raw features)\n', ...
    k95, numel(eigVals));

Vk        = V(:, 1:k95);
XTrainPCA = XTrainNorm * Vk;
XTestPCA  = XTestNorm  * Vk;

% 4.7 Scree plot and cumulative variance curve
figure('Position',[100 100 1000 420]);
subplot(1,2,1);
nShowPlot = min(60, numel(eigVals));
plot(1:nShowPlot, eigVals(1:nShowPlot),'b-o','LineWidth',2,'MarkerSize',5);
xlabel('Principal Component'); ylabel('Eigenvalue');
title('Scree Plot - Top 60 PCA Components'); grid on;

subplot(1,2,2);
plot(1:numel(cumVar), cumVar*100,'r-','LineWidth',2); hold on;
plot([k95 k95],[0 100],'k--','LineWidth',1.5);
plot([0 numel(cumVar)],[95 95],'g--','LineWidth',1.5);
xlabel('Number of Components'); ylabel('Cumulative Variance (%)');
title('PCA Cumulative Variance Explained');
legend('Cumulative Var',['k=' num2str(k95)],'95% threshold','Location','southeast');
grid on; hold off;

% 4.8 3D PCA scatter on PC1-PC3 with class centroids
yTrainCat = categorical(trainLabels);
classN    = categories(yTrainCat);
pcColors  = {[0.2 0.4 0.8],[0.8 0.2 0.2]};
figure('Position',[100 100 700 600]); hold on;
for ci = 1:numel(classN)
    idx = (yTrainCat == classN{ci});
    scatter3(XTrainPCA(idx,1),XTrainPCA(idx,2),XTrainPCA(idx,3), ...
        12,pcColors{ci},'filled','MarkerFaceAlpha',0.3,'DisplayName',classN{ci});
end
for ci = 1:numel(classN)
    idx = (yTrainCat == classN{ci});
    ns_ = sum(idx);
    mx = sum(XTrainPCA(idx,1))/ns_;
    my = sum(XTrainPCA(idx,2))/ns_;
    mz = sum(XTrainPCA(idx,3))/ns_;
    scatter3(mx,my,mz,250,pcColors{ci},'filled','p','MarkerEdgeColor','k', ...
        'LineWidth',2,'DisplayName',[classN{ci} ' centroid']);
end
xlabel('PC 1'); ylabel('PC 2'); zlabel('PC 3');
title('3D PCA Scatter - PC1 vs PC2 vs PC3 (star = class centroid)');
legend; grid on; view(45,30); hold off;

save('features_data.mat', ...
    'XTrainPCA','XTestPCA','XTrainNorm','XTestNorm', ...
    'trainLabels','testLabels','Vk','featMean','featStd', ...
    'eigVals','cumVar','k95');
fprintf('\nStep 4 - Feature Engineering Complete.\n');


%% =====================================================================
%  STEP 5 - STATISTICAL ANALYSIS (descriptive + inferential)
% ======================================================================
load('features_data.mat');

yBin     = strcmp(trainLabels,'DR');
masks    = {~logical(yBin), logical(yBin)};
classNms = {'No_DR','DR'};

fprintf('\n=== STEP 5: STATISTICAL ANALYSIS ===\n');
fprintf('Training samples: %d  (DR=%d, No_DR=%d)\n', ...
    numel(yBin), sum(yBin), sum(~yBin));

% --- A. Descriptive statistics: PC1 to PC5 ---
fprintf('\n--- Descriptive Statistics: PC1 to PC5 ---\n');
fprintf('%-8s | %-10s | %-10s | %-10s | %-12s | %-10s\n', ...
    'Feature','Mean','Median','Std','Skewness','Kurtosis');
fprintf('%s\n', repmat('-',1,68));
for fi = 1:5
    x  = XTrainPCA(:,fi); n_ = numel(x);
    mn = sum(x)/n_;
    xs = sort(x);  med = xs(round(0.5*n_));
    dev= x - mn;   sd  = sqrt(sum(dev.^2)/(n_-1));
    sk = (sum(dev.^3)/n_)/sd^3;
    kt = (sum(dev.^4)/n_)/sd^4;
    fprintf('%-8s | %-10.4f | %-10.4f | %-10.4f | %-12.4f | %-10.4f\n', ...
        ['PC' num2str(fi)],mn,med,sd,sk,kt);
end

% --- B. Class-wise stats PC1 to PC3 ---
fprintf('\n--- Class-wise Mean / Std / Median: PC1 to PC3 ---\n');
fprintf('%-8s | %-5s | %-10s | %-10s | %-10s\n','Class','PC','Mean','Std','Median');
fprintf('%s\n', repmat('-',1,55));
for ci = 1:2
    for fi = 1:3
        x  = XTrainPCA(masks{ci},fi); n_=numel(x);
        mn = sum(x)/n_;
        xs = sort(x);  med = xs(round(0.5*n_));
        dev= x-mn;     sd  = sqrt(sum(dev.^2)/(n_-1));
        fprintf('%-8s | %-5s | %-10.4f | %-10.4f | %-10.4f\n', ...
            classNms{ci},['PC' num2str(fi)],mn,sd,med);
    end
    if ci==1, fprintf('%s\n',repmat('-',1,55)); end
end

% --- C. Welch two-sample t-test on PC1-PC5 ---
t_crit  = 1.96;
tStatsW = zeros(1,5);
fprintf('\n--- Welch t-Test: DR vs No_DR (PC1-PC5) ---\n');
fprintf('%-8s | %-10s | %-12s | %-12s | %-10s | %-10s\n', ...
    'Feature','Mean DR','Mean NoDR','t-stat','df','Reject H0?');
fprintf('%s\n',repmat('-',1,72));
for fi = 1:5
    x1=XTrainPCA(logical(yBin),fi);  x2=XTrainPCA(~logical(yBin),fi);
    n1=numel(x1); n2=numel(x2);
    m1=sum(x1)/n1; m2=sum(x2)/n2;
    s1=sqrt(sum((x1-m1).^2)/(n1-1)); s2=sqrt(sum((x2-m2).^2)/(n2-1));
    se=sqrt(s1^2/n1+s2^2/n2);
    t =(m1-m2)/(se+1e-9);
    df=(s1^2/n1+s2^2/n2)^2/((s1^2/n1)^2/(n1-1)+(s2^2/n2)^2/(n2-1));
    tStatsW(fi)=t;
    if abs(t)>t_crit, rej='Yes'; else, rej='No'; end
    fprintf('%-8s | %-10.4f | %-12.4f | %-12.4f | %-10.1f | %-10s\n', ...
        ['PC' num2str(fi)],m1,m2,t,df,rej);
end

% --- D. Mann-Whitney U test on PC1 ---
x1 = XTrainPCA(logical(yBin),1);   x2 = XTrainPCA(~logical(yBin),1);
n1 = numel(x1);                    n2 = numel(x2);
combined = [x1;x2]; group = [ones(n1,1);zeros(n2,1)];
[~,sortI] = sort(combined);
ranks = zeros(numel(combined),1);
for ri = 1:numel(sortI), ranks(sortI(ri)) = ri; end
R1    = sum(ranks(group==1));
U1    = R1 - n1*(n1+1)/2;
U2    = n1*n2 - U1;
U     = min(U1,U2);
mu_U  = n1*n2/2;
sig_U = sqrt(n1*n2*(n1+n2+1)/12);
z_U   = (U-mu_U)/sig_U;
fprintf('\n--- Mann-Whitney U Test (PC1) ---\n');
fprintf('U=%.1f  |  z=%.4f  |  Significant (|z|>1.96): %s\n', ...
    U, z_U, string(abs(z_U)>1.96));

% --- E. 95% confidence intervals on PC1 means ---
fprintf('\n--- 95%% Confidence Intervals - PC1 ---\n');
fprintf('%-8s | %-10s | %-12s | %-12s\n','Class','Mean','CI Lower','CI Upper');
for ci = 1:2
    x  = XTrainPCA(masks{ci},1); n_ = numel(x);
    mn = sum(x)/n_;
    dev= x - mn; sd = sqrt(sum(dev.^2)/(n_-1)); se = sd/sqrt(n_);
    fprintf('%-8s | %-10.4f | %-12.4f | %-12.4f\n', ...
        classNms{ci}, mn, mn-1.96*se, mn+1.96*se);
end

% --- F. Pearson correlation matrix of PC1-PC10 ---
nC   = min(10,size(XTrainPCA,2));
Xcor = XTrainPCA(:,1:nC);
corrMat = zeros(nC,nC);
for i = 1:nC
    for j = 1:nC
        xi = Xcor(:,i); xj = Xcor(:,j);
        n_ij = numel(xi);
        mi = sum(xi)/n_ij;  mj = sum(xj)/n_ij;
        cov_ij = sum((xi-mi).*(xj-mj))/(n_ij-1);
        si = sqrt(sum((xi-mi).^2)/(n_ij-1));
        sj = sqrt(sum((xj-mj).^2)/(n_ij-1));
        if si*sj < 1e-10, corrMat(i,j) = 0;
        else,             corrMat(i,j) = cov_ij/(si*sj); end
    end
end
fprintf('\nPearson Correlation Matrix (PC1-PC10):\n');
disp(round(corrMat,3));

% Welch t-statistic bar chart
figure('Position',[100 100 650 400]);
bar(1:5,tStatsW,'FaceColor',[0.3 0.7 0.4]); hold on;
plot([0.5 5.5],[t_crit t_crit],'r--','LineWidth',2);
plot([0.5 5.5],[-t_crit -t_crit],'r--','LineWidth',2);
set(gca,'XTick',1:5,'XTickLabel',{'PC1','PC2','PC3','PC4','PC5'});
xlabel('PCA Feature'); ylabel('t-statistic');
title('Welch t-Test - DR vs No\_DR (dashed = +/-1.96 critical value)');
legend('t-statistic','+/-1.96 Critical','Location','best'); grid on; hold off;

[~, bestPC] = max(abs(tStatsW));
fprintf('\nKey finding: PC%d has the largest |t| = %.2f\n', bestPC, abs(tStatsW(bestPC)));
fprintf('Step 5 - Statistical Analysis Complete.\n');


%% =====================================================================
%  STEP 6: 3D FEATURE SPACE OF DIABETIC RETINOPATHY CLASSES
%  ---------------------------------------------------------------------
%  Reference figure showing all 5 original severity classes in one 3D
%  plot using raw image descriptors (Brightness, Contrast, Sharpness).
%
%  Each class is represented by:
%    - A coloured sphere positioned at the class MEAN (Brightness,
%      Contrast, Sharpness)
%    - Sphere SIZE proportional to overall class VARIANCE (spread)
%    - Error bars extending +/-1 STD along each of the 3 axes
%    - A horizontal reference plane at the centroid's sharpness level
%    - A text annotation with the class name and Brightness skewness (Sk)
%
%  Visual encoding:
%    Position  = Mean (central tendency)
%    Size      = Variance (class spread)
%    Colour    = Severity class identity
%    Error bars= +/-1 Std (distributional uncertainty)
%    Planes    = Reference sharpness level per class
% ======================================================================

% Restore 5-class imageDatastore labels for this figure
%   (imds was converted to binary in Step 2; re-derive 5-class indices
%    from the original brightnessValues / contrastValues / sharpnessValues
%    which were computed before the binary conversion in Step 1)
load('binary_basic_features.mat', ...
    'brightnessValues','contrastValues','sharpnessValues');

% Re-load the 5-class datastore (read-only; does not overwrite imds)
imds5 = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource',       'foldernames');

% Colour palette — legend order matches reference figure:
%   Mild (blue/teal) | Moderate (red) | No_DR (green/teal) |
%   Proliferate_DR (orange) | Severe (pink/lilac)
classPalette = [0.20 0.55 0.75;    % Mild          - blue/teal
                0.80 0.25 0.25;    % Moderate      - red
                0.15 0.65 0.50;    % No_DR         - green/teal
                0.95 0.55 0.15;    % Proliferate   - orange
                0.85 0.55 0.80];   % Severe        - pink/lilac

classNamesFig28 = {'Mild','Moderate','No_DR','Proliferate_DR','Severe'};

% Pre-compute per-class statistics:
%   muB, muC, muS  = mean of each feature
%   sdB, sdC, sdS  = std dev along each feature (for error bars)
%   varTotal       = combined variance across 3 features (for sphere size)
%   skB            = skewness of Brightness channel (shown as annotation)
stats28 = struct();
for c = 1:numel(classNamesFig28)
    idxC = find(imds5.Labels == classNamesFig28{c});
    bV = brightnessValues(idxC);
    cV = contrastValues(idxC);
    sV = sharpnessValues(idxC);
    nC = numel(idxC);

    stats28(c).name = classNamesFig28{c};
    stats28(c).n    = nC;
    stats28(c).muB  = sum(bV)/nC;
    stats28(c).muC  = sum(cV)/nC;
    stats28(c).muS  = sum(sV)/nC;
    stats28(c).sdB  = sqrt(sum((bV - stats28(c).muB).^2)/(nC-1));
    stats28(c).sdC  = sqrt(sum((cV - stats28(c).muC).^2)/(nC-1));
    stats28(c).sdS  = sqrt(sum((sV - stats28(c).muS).^2)/(nC-1));

    % Total variance used for marker-size encoding (normalised later)
    stats28(c).varTotal = stats28(c).sdB^2 + stats28(c).sdC^2 + ...
                          (stats28(c).sdS/500)^2;

    % Skewness of Brightness channel
    devB = bV - stats28(c).muB;
    stats28(c).skB = (sum(devB.^3)/nC) / (stats28(c).sdB^3 + 1e-9);
end

% --- Figure construction ------------------------------------------------
figure('Position', [60 60 1100 780], 'Color', 'white');
hold on;

% Normalise variance -> marker size in the range [180, 520]
vAll = [stats28.varTotal];
vMin = min(vAll); vMax = max(vAll);
sizeMap = @(v) 180 + 340 * (v - vMin) / (vMax - vMin + 1e-9);

for c = 1:numel(classNamesFig28)
    S   = stats28(c);
    col = classPalette(c,:);
    mSz = sizeMap(S.varTotal);

    % --- Horizontal reference plane at the class's sharpness level -----
    % Plane spans the brightness x contrast extent of the plot
    xB = [134.8, 137.2];    % brightness range of the plot
    yC = [13.5,  24.5];     % contrast   range of the plot
    [pxB, pyC] = meshgrid(xB, yC);
    pzS = S.muS * ones(size(pxB));
    surf(pxB, pyC, pzS, ...
        'FaceColor', col, 'FaceAlpha', 0.08, ...
        'EdgeColor', col, 'EdgeAlpha', 0.25, ...
        'LineStyle', '-', 'HandleVisibility', 'off');

    % --- Error bars (+/-1 SD) along each of the three axes -------------
    % Brightness axis
    plot3([S.muB-S.sdB, S.muB+S.sdB], [S.muC, S.muC], [S.muS, S.muS], ...
        '-', 'Color', col, 'LineWidth', 1.8, 'HandleVisibility', 'off');
    plot3(S.muB-S.sdB, S.muC, S.muS, '+', 'Color', col, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    plot3(S.muB+S.sdB, S.muC, S.muS, '+', 'Color', col, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility', 'off');

    % Contrast axis
    plot3([S.muB, S.muB], [S.muC-S.sdC, S.muC+S.sdC], [S.muS, S.muS], ...
        '-', 'Color', col, 'LineWidth', 1.8, 'HandleVisibility', 'off');
    plot3(S.muB, S.muC-S.sdC, S.muS, '+', 'Color', col, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    plot3(S.muB, S.muC+S.sdC, S.muS, '+', 'Color', col, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility', 'off');

    % Sharpness axis
    plot3([S.muB, S.muB], [S.muC, S.muC], [S.muS-S.sdS, S.muS+S.sdS], ...
        '-', 'Color', col, 'LineWidth', 1.8, 'HandleVisibility', 'off');
    plot3(S.muB, S.muC, S.muS-S.sdS, '+', 'Color', col, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    plot3(S.muB, S.muC, S.muS+S.sdS, '+', 'Color', col, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility', 'off');

    % --- Centroid sphere (size encodes class variance) -----------------
    scatter3(S.muB, S.muC, S.muS, mSz, col, 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.6, ...
        'DisplayName', strrep(S.name, '_', '\_'));

    % --- Class label + skewness annotation -----------------------------
    txt = sprintf('%s\nSk=%.2f', strrep(S.name,'_','_{D}'), S.skB);
    text(S.muB + 0.08, S.muC + 0.4, S.muS + 70, txt, ...
        'FontSize', 9, 'FontWeight', 'bold', 'Color', col, ...
        'BackgroundColor', 'white', 'EdgeColor', col, ...
        'Margin', 2, 'HandleVisibility', 'off');
end

% --- Axes cosmetics -----------------------------------------------------
xlabel('Brightness (Mean)',  'FontSize', 12, 'FontWeight', 'bold');
ylabel('Contrast (Mean)',    'FontSize', 12, 'FontWeight', 'bold');
zlabel('Sharpness (Mean)',   'FontSize', 12, 'FontWeight', 'bold');
title({'3D Feature Space of Diabetic Retinopathy Classes', ...
       'Position = Mean | Size = Variance | Colour = Class | Error Bars = \pm1 Std'}, ...
       'FontSize', 13, 'FontWeight', 'bold');

legend('Location', 'northeastoutside', 'FontSize', 10);
grid on; box on;
view(-38, 22);
ax = gca;
ax.FontSize      = 11;
ax.GridAlpha     = 0.30;
ax.GridLineStyle = '--';
ax.LineWidth     = 1.0;

% Fix axis limits so the reference planes fit cleanly
xlim([134.8, 137.2]);
ylim([13.5,  24.5]);
zlim([2300, 3700]);
hold off;

% --- Console summary matching the figure caption ----------------------
fprintf('\n=== 3D FEATURE SPACE: CLASS SUMMARY ===\n');
fprintf('%-15s | %-5s | %-20s | %-20s | %-20s\n', ...
    'Class','n','Brightness (mu+/-sd)','Contrast (mu+/-sd)','Sharpness (mu+/-sd)');
fprintf('%s\n', repmat('-',1,88));
for c = 1:numel(classNamesFig28)
    S = stats28(c);
    fprintf('%-15s | %-5d | %8.1f +/- %6.1f   | %8.1f +/- %6.1f   | %8.0f +/- %6.0f\n', ...
        S.name, S.n, S.muB, S.sdB, S.muC, S.sdC, S.muS, S.sdS);
end
fprintf('\nStep 6 - Multi-Information Visualisation Complete\n');


%% =====================================================================
%  STEP 7 - MACHINE LEARNING CLASSIFICATION
% ======================================================================
load('features_data.mat');
yTrain = double(strcmp(trainLabels,'DR'));
yTest  = double(strcmp(testLabels,'DR'));
XTr    = XTrainPCA;  XTe = XTestPCA;
nTrain = size(XTr,1); nTest = size(XTe,1);

fprintf('\n=== STEP 7: MACHINE LEARNING ===\n');
fprintf('Training: %d  (DR=%d, No_DR=%d)\n',nTrain,sum(yTrain),sum(yTrain==0));
fprintf('Testing:  %d  (DR=%d, No_DR=%d)\n',nTest, sum(yTest), sum(yTest==0));

% 7.1 KNN (k=5)
fprintf('\nTraining KNN (k=5)...\n');
predKNN   = manualKNN(XTr, yTrain, XTe, 5);
knnScores = zeros(nTest,1);
for i = 1:nTest
    diffs = XTr - repmat(XTe(i,:),nTrain,1);
    dists = sqrt(sum(diffs.^2,2));
    [~,sI] = sort(dists);
    knnScores(i) = sum(yTrain(sI(1:5)))/5;
end

% 7.2 Linear SVM via SGD
fprintf('\nTraining Linear SVM (200 epochs, C=1.0, lr=0.001)...\n');
rng(42);
[wSVM,bSVM,svmLossHistory] = trainLinearSVM(XTr, yTrain, 1.0, 0.001, 200);
scoresSVM = XTe*wSVM + bSVM;
predSVM   = double(scoresSVM >= 0);

% 7.3 Decision Tree (Gini, depth 5)
fprintf('\nTraining Decision Tree (Gini, depth=5)...\n');
rng(42);
tree   = buildTree(XTr, yTrain, 0, 5);
predDT = predictTree(tree, XTe);

% 7.4 Gaussian Naive Bayes
fprintf('\nTraining Gaussian Naive Bayes...\n');
[predNB, scoresNB] = gaussianNaiveBayes(XTr, yTrain, XTe);

% 7.5 Random Forest (10 trees)
fprintf('\nTraining Random Forest (10 trees, bootstrap, sqrt features)...\n');
rng(42);
nTrees = 10;
forest = cell(nTrees,1);
for t = 1:nTrees
    bootIdx   = randi(nTrain, nTrain, 1);
    forest{t} = buildTree(XTr(bootIdx,:), yTrain(bootIdx), 0, 8);
    if mod(t,5)==0, fprintf('  Tree %d/%d built\n',t,nTrees); end
end
predRF   = predictForest(forest, XTe);
scoresRF = predictForestProb(forest, XTe);

% 7.6 Evaluation
[accKNN,sensKNN,specKNN,f1KNN,aucKNN] = evalMetrics(yTest,predKNN,knnScores);
[accSVM,sensSVM,specSVM,f1SVM,aucSVM] = evalMetrics(yTest,predSVM,scoresSVM);
[accDT, sensDT, specDT, f1DT, aucDT]  = evalMetrics(yTest,predDT,[]);
[accNB, sensNB, specNB, f1NB, aucNB]  = evalMetrics(yTest,predNB,scoresNB);
[accRF, sensRF, specRF, f1RF, aucRF]  = evalMetrics(yTest,predRF,scoresRF);

algoNames = {'KNN (k=5)','Linear SVM','Decision Tree','Naive Bayes','Random Forest'};

fprintf('\n====== EVALUATION RESULTS ======\n');
fprintf('%-15s %-10s %-12s %-12s %-10s %-10s\n', ...
    'Algorithm','Accuracy','Sensitivity','Specificity','F1-Score','AUC');
fprintf('%s\n',repmat('-',1,64));
fprintf('%-15s %-10.4f %-12.4f %-12.4f %-10.4f %-10.4f\n','KNN (k=5)',    accKNN,sensKNN,specKNN,f1KNN,aucKNN);
fprintf('%-15s %-10.4f %-12.4f %-12.4f %-10.4f %-10.4f\n','Linear SVM',   accSVM,sensSVM,specSVM,f1SVM,aucSVM);
fprintf('%-15s %-10.4f %-12.4f %-12.4f %-10.4f %-10.4f\n','Decision Tree',accDT, sensDT, specDT, f1DT, aucDT);
fprintf('%-15s %-10.4f %-12.4f %-12.4f %-10.4f %-10.4f\n','Naive Bayes',  accNB, sensNB, specNB, f1NB, aucNB);
fprintf('%-15s %-10.4f %-12.4f %-12.4f %-10.4f %-10.4f\n','Random Forest',accRF, sensRF, specRF, f1RF, aucRF);

% 7.7 Confusion matrices
CMs = {confusionMat(yTest,predKNN), confusionMat(yTest,predSVM), ...
       confusionMat(yTest,predDT),  confusionMat(yTest,predNB), ...
       confusionMat(yTest,predRF)};
blueMap = [linspace(1,0.1,64)', linspace(1,0.3,64)', linspace(1,0.8,64)'];
figure('Position',[50 50 1600 360]);
for ai = 1:5
    subplot(1,5,ai); CM = CMs{ai};
    imagesc(CM); colormap(blueMap); clim([0 max(CM(:))]);
    for r = 1:2
        for c = 1:2
            text(c,r,num2str(CM(r,c)),'HorizontalAlignment','center', ...
                'FontSize',14,'FontWeight','bold','Color','k');
        end
    end
    set(gca,'XTick',[1 2],'XTickLabel',{'No\_DR','DR'}, ...
            'YTick',[1 2],'YTickLabel',{'No\_DR','DR'},'FontSize',9);
    xlabel('Predicted'); ylabel('Actual');
    title(['CM: ' algoNames{ai}]); colorbar;
end

% 7.8 ROC curves
figure('Position',[50 50 750 600]); hold on;
scoresList = {knnScores, scoresSVM, scoresNB, scoresRF};
aucs       = {aucKNN,    aucSVM,    aucNB,    aucRF};
rocNames   = {'KNN','SVM','Naive Bayes','Random Forest'};
rocColors  = {'b','r','m','g'};
for ri = 1:4
    sc   = scoresList{ri};
    thrs = sort(unique(sc),'descend');
    tprR = zeros(numel(thrs)+2,1); fprR = zeros(numel(thrs)+2,1);
    for ti = 1:numel(thrs)
        pT  = double(sc>=thrs(ti));
        tp_ = sum(yTest==1 & pT==1); fp_ = sum(yTest==0 & pT==1);
        fn_ = sum(yTest==1 & pT==0); tn_ = sum(yTest==0 & pT==0);
        tprR(ti+1) = tp_/(tp_+fn_+1e-9);
        fprR(ti+1) = fp_/(fp_+tn_+1e-9);
    end
    tprR(end) = 1; fprR(end) = 1;
    plot(fprR,tprR,rocColors{ri},'LineWidth',2, ...
        'DisplayName',sprintf('%s AUC=%.3f',rocNames{ri},aucs{ri}));
end
plot([0 1],[0 1],'k--','LineWidth',1,'DisplayName','Random (AUC=0.5)');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves - All Classifiers'); legend('Location','southeast'); grid on; hold off;

% 7.9 Grouped metric bar chart
metrics = [accKNN  accSVM  accDT  accNB  accRF;
           sensKNN sensSVM sensDT sensNB sensRF;
           specKNN specSVM specDT specNB specRF;
           f1KNN   f1SVM   f1DT   f1NB   f1RF;
           aucKNN  aucSVM  aucDT  aucNB  aucRF];
metNames = {'Accuracy','Sensitivity','Specificity','F1-Score','AUC'};
figure('Position',[50 50 1100 500]);
bh = bar(metrics,'grouped');
algoCols = {[0.2 0.4 0.8],[0.8 0.2 0.2],[0.2 0.7 0.3],[0.9 0.5 0.1],[0.5 0.1 0.7]};
for ai = 1:5, bh(ai).FaceColor = algoCols{ai}; end
set(gca,'XTick',1:5,'XTickLabel',metNames,'FontSize',10);
ylabel('Score'); ylim([0 1]);
title('Algorithm Comparison - KNN | SVM | Decision Tree | Naive Bayes | Random Forest');
legend(algoNames,'Location','southeast'); grid on;

% 7.10 KNN hyperparameter study
kVals   = [1,3,5,7,9,11,15];
accVals = zeros(1,numel(kVals)); f1Vals = zeros(1,numel(kVals));
fprintf('\n--- KNN Hyperparameter Analysis ---\n');
fprintf('%-5s %-10s %-10s\n','k','Accuracy','F1-Score');
for ki = 1:numel(kVals)
    pK = manualKNN(XTr,yTrain,XTe,kVals(ki));
    [a,~,~,f,~] = evalMetrics(yTest,pK,[]);
    accVals(ki)=a; f1Vals(ki)=f;
    fprintf('%-5d %-10.4f %-10.4f\n',kVals(ki),a,f);
end
[bestAcc,bestAccIdx] = max(accVals);
[bestF1, bestF1Idx ] = max(f1Vals);
figure('Position',[50 50 900 420]);
subplot(1,2,1);
plot(kVals,accVals*100,'b-o','LineWidth',2,'MarkerSize',8); hold on;
plot(kVals(bestAccIdx),accVals(bestAccIdx)*100,'r*','MarkerSize',14,'LineWidth',2, ...
    'DisplayName',['Best k=' num2str(kVals(bestAccIdx))]);
xlabel('k Value'); ylabel('Accuracy (%)'); title('KNN Accuracy vs k');
legend('Location','best'); grid on; hold off;
subplot(1,2,2);
plot(kVals,f1Vals,'r-o','LineWidth',2,'MarkerSize',8); hold on;
plot(kVals(bestF1Idx),f1Vals(bestF1Idx),'b*','MarkerSize',14,'LineWidth',2, ...
    'DisplayName',['Best k=' num2str(kVals(bestF1Idx))]);
xlabel('k Value'); ylabel('F1-Score'); title('KNN F1-Score vs k');
legend('Location','best'); grid on; hold off;
fprintf('\nBest k by Accuracy: k=%d (%.4f)\n',kVals(bestAccIdx),bestAcc);
fprintf('Best k by F1-Score: k=%d (%.4f)\n',kVals(bestF1Idx),bestF1);

% 7.11 SVM convergence
figure('Position',[50 50 700 400]);
plot(1:numel(svmLossHistory), svmLossHistory,'r-','LineWidth',2);
xlabel('Epoch'); ylabel('Hinge Loss');
title('SVM Training Convergence - Hinge Loss per Epoch'); grid on;

% 7.12 Final summary
fprintf('\n====== FINAL SUMMARY ======\n');
allAcc  = [accKNN  accSVM  accDT  accNB  accRF];
allAUC  = [aucKNN  aucSVM  aucDT  aucNB  aucRF];
allF1   = [f1KNN   f1SVM   f1DT   f1NB   f1RF];
allSens = [sensKNN sensSVM sensDT sensNB sensRF];
[~,bi] = max(allAcc);  fprintf('Best Accuracy    : %s  (%.4f)\n',algoNames{bi},max(allAcc));
[~,bi] = max(allAUC);  fprintf('Best AUC         : %s  (%.4f)\n',algoNames{bi},max(allAUC));
[~,bi] = max(allF1);   fprintf('Best F1-Score    : %s  (%.4f)\n',algoNames{bi},max(allF1));
[~,bi] = max(allSens); fprintf('Best Sensitivity : %s  (%.4f)\n',algoNames{bi},max(allSens));
fprintf('\nStep 7 - Machine Learning Complete.\n');


%% =====================================================================
%  LOCAL FUNCTIONS
% ======================================================================

% --- Preprocessing: grayscale, contrast stretch, resize --------------
function procImg = manualPreprocess(img)
    img = double(img);
    if size(img,3)==3
        g = 0.2989*img(:,:,1) + 0.5870*img(:,:,2) + 0.1140*img(:,:,3);
    else
        g = img;
    end
    mn = min(g(:)); mx = max(g(:));
    if mx>mn, g = (g-mn)/(mx-mn); else, g = zeros(size(g)); end
    sorted = sort(g(:)); n = numel(sorted);
    lo = sorted(max(1,round(0.02*n)));
    hi = sorted(min(n,round(0.98*n)));
    if hi>lo, g = (g-lo)/(hi-lo); g(g<0)=0; g(g>1)=1; end
    tSz = 64;
    [rO,cO] = size(g);
    procImg = zeros(tSz,tSz);
    for r = 1:tSz
        for cc = 1:tSz
            sR = (r-1)*(rO-1)/(tSz-1) + 1;
            sC = (cc-1)*(cO-1)/(tSz-1) + 1;
            r1 = floor(sR); r2 = min(r1+1,rO);
            c1 = floor(sC); c2 = min(c1+1,cO);
            dr = sR-r1;    dc = sC-c1;
            procImg(r,cc) = (1-dr)*(1-dc)*g(r1,c1) + (1-dr)*dc*g(r1,c2) ...
                           +    dr *(1-dc)*g(r2,c1) +    dr *dc*g(r2,c2);
        end
    end
end

% --- Augmentation helpers --------------------------------------------
function aug = flipHorizontal(img), aug = img(:,   end:-1:1); end
function aug = flipVertical(  img), aug = img(end:-1:1,  :); end

function aug = rotateManual(img, angleDeg)
    [rows,cols] = size(img);
    aug = zeros(rows,cols);
    cx = (cols+1)/2; cy = (rows+1)/2;
    theta = angleDeg*pi/180;
    cosT = cos(-theta); sinT = sin(-theta);
    for r = 1:rows
        for c = 1:cols
            x = c-cx; y = r-cy;
            srcX = cosT*x - sinT*y + cx;
            srcY = sinT*x + cosT*y + cy;
            if srcX>=1 && srcX<=cols && srcY>=1 && srcY<=rows
                r1 = floor(srcY); r2 = min(r1+1,rows);
                c1 = floor(srcX); c2 = min(c1+1,cols);
                dr = srcY-r1;     dc = srcX-c1;
                aug(r,c) = (1-dr)*(1-dc)*img(r1,c1) + (1-dr)*dc*img(r1,c2) ...
                          +    dr *(1-dc)*img(r2,c1) +    dr *dc*img(r2,c2);
            end
        end
    end
end

function aug = addGaussianNoise(img, sigma)
    [r,c] = size(img);
    U1 = rand(r,c); U1(U1<1e-10) = 1e-10;
    U2 = rand(r,c);
    noise = sqrt(-2*log(U1)).*cos(2*pi*U2);
    aug = img + sigma*noise;
    aug(aug<0) = 0; aug(aug>1) = 1;
end

% --- HOG -------------------------------------------------------------
function hogFeat = computeHOG(img, cellSize, numBins)
    [rows,cols] = size(img);
    Gx = zeros(rows,cols); Gy = zeros(rows,cols);
    for r = 2:rows-1
        for c = 2:cols-1
            Gx(r,c) = img(r,c+1) - img(r,c-1);
            Gy(r,c) = img(r+1,c) - img(r-1,c);
        end
    end
    mag   = sqrt(Gx.^2 + Gy.^2);
    angle = atan2(Gy,Gx)*(180/pi);
    angle(angle<0) = angle(angle<0) + 180;

    nCellsR = floor(rows/cellSize);
    nCellsC = floor(cols/cellSize);
    hogFeat = zeros(1, nCellsR*nCellsC*numBins);
    featIdx = 1;
    binEdges = linspace(0,180,numBins+1);

    for cr = 1:nCellsR
        for cc = 1:nCellsC
            r1 = (cr-1)*cellSize+1; r2 = cr*cellSize;
            c1 = (cc-1)*cellSize+1; c2 = cc*cellSize;
            cellMag   = mag(r1:r2, c1:c2);
            cellAngle = angle(r1:r2, c1:c2);
            h = zeros(1,numBins);
            for b = 1:numBins
                mask = (cellAngle>=binEdges(b)) & (cellAngle<binEdges(b+1));
                h(b) = sum(cellMag(mask));
            end
            h = h / (sqrt(sum(h.^2)) + 1e-6);
            hogFeat(featIdx:featIdx+numBins-1) = h;
            featIdx = featIdx + numBins;
        end
    end
end

% --- LBP -------------------------------------------------------------
function lbpFeat = computeLBP(img)
    [rows,cols] = size(img);
    lbpImg = zeros(rows,cols);
    offsets = [-1,-1;-1,0;-1,1; 0,1; 1,1; 1,0; 1,-1; 0,-1];
    for r = 2:rows-1
        for c = 2:cols-1
            centerVal = img(r,c);
            code = 0;
            for k = 1:8
                nr = r+offsets(k,1); nc = c+offsets(k,2);
                if img(nr,nc) >= centerVal
                    code = code + 2^(k-1);
                end
            end
            lbpImg(r,c) = code;
        end
    end
    lbpFeat = zeros(1,256);
    for v = 0:255
        lbpFeat(v+1) = sum(lbpImg(:)==v);
    end
    lbpFeat = lbpFeat / (sum(lbpFeat) + 1e-6);
end

% --- Gabor -----------------------------------------------------------
function gaborFeat = computeGabor(img)
    freqs  = [0.1, 0.2];
    thetas = [0, 45, 90, 135]*pi/180;
    sigma  = 3;  gamma = 0.5;  kSize = 15; halfK = floor(kSize/2);
    gaborFeat = [];
    for fi = 1:numel(freqs)
        f = freqs(fi);
        for ti = 1:numel(thetas)
            theta = thetas(ti);
            kernel = zeros(kSize,kSize);
            for kr = 1:kSize
                for kc = 1:kSize
                    x = kc - halfK - 1;
                    y = -(kr - halfK - 1);
                    xp =  x*cos(theta) + y*sin(theta);
                    yp = -x*sin(theta) + y*cos(theta);
                    kernel(kr,kc) = exp(-(xp^2 + gamma^2*yp^2)/(2*sigma^2)) ...
                                    * cos(2*pi*f*xp);
                end
            end
            response = abs(conv2(img,kernel,'same'));
            n_resp = numel(response);
            mn_ = sum(response(:))/n_resp;
            st_ = sqrt(sum((response(:)-mn_).^2)/n_resp);
            gaborFeat = [gaborFeat, mn_, st_]; %#ok<AGROW>
        end
    end
end

% --- KNN -------------------------------------------------------------
function predLabels = manualKNN(XTrain, yTrain, XTest, k)
    nTest_ = size(XTest,1);
    predLabels = zeros(nTest_,1);
    for i = 1:nTest_
        diffs = XTrain - repmat(XTest(i,:), size(XTrain,1), 1);
        dists = sqrt(sum(diffs.^2, 2));
        [~,sortI] = sort(dists);
        predLabels(i) = round(sum(yTrain(sortI(1:k)))/k);
    end
end

% --- Linear SVM via SGD ----------------------------------------------
function [w, b, lossHistory] = trainLinearSVM(X, y, C, lr, epochs)
    [n,d] = size(X);
    w = zeros(d,1); b = 0;
    y_ = 2*y - 1;
    lossHistory = zeros(epochs,1);
    for ep = 1:epochs
        idx = randperm(n); X_ = X(idx,:); y__ = y_(idx);
        for i = 1:n
            xi = X_(i,:)'; yi = y__(i);
            if yi*(w'*xi + b) < 1
                w = w - lr*(w/n - C*yi*xi);
                b = b + lr*C*yi;
            else
                w = w - lr*(w/n);
            end
        end
        scores_ = X_*w + b;
        lossHistory(ep) = sum(max(0, 1 - y__ .* scores_)) / n;
        if mod(ep,50)==0
            fprintf('  Epoch %d: hinge loss = %.4f\n', ep, lossHistory(ep));
        end
    end
end

% --- Gaussian Naive Bayes --------------------------------------------
function [predictions, scores] = gaussianNaiveBayes(XTrain, yTrain, XTest)
    classes  = unique(yTrain);
    nClasses = numel(classes);
    nFeat    = size(XTrain,2);
    nTest    = size(XTest,1);
    classMeans = zeros(nClasses, nFeat);
    classVars  = zeros(nClasses, nFeat);
    classPrior = zeros(nClasses, 1);
    for ci = 1:nClasses
        idx = (yTrain == classes(ci));
        classPrior(ci)   = sum(idx)/numel(yTrain);
        classMeans(ci,:) = sum(XTrain(idx,:),1)/sum(idx);
        for j = 1:nFeat
            dev = XTrain(idx,j) - classMeans(ci,j);
            classVars(ci,j) = max(sum(dev.^2)/sum(idx), 1e-9);
        end
    end
    logPost = zeros(nTest, nClasses);
    for ci = 1:nClasses
        lp = log(classPrior(ci));
        for j = 1:nFeat
            mu_  = classMeans(ci,j);
            sig2 = classVars(ci,j);
            lp = lp + (-0.5*log(2*pi*sig2)) ...
                    + (-(XTest(:,j)-mu_).^2 / (2*sig2));
        end
        logPost(:,ci) = lp;
    end
    [~,argmax] = max(logPost,[],2);
    predictions = classes(argmax);
    posIdx = find(classes==1);
    expLP  = exp(logPost - max(logPost,[],2));
    scores = expLP(:,posIdx) ./ sum(expLP,2);
end

% --- Decision Tree helpers -------------------------------------------
function g = giniImpurity(y)
    n = numel(y);
    if n==0, g = 0; return; end
    p1 = sum(y==1)/n;
    g  = 1 - p1^2 - (1-p1)^2;
end

function [bFeat,bThresh,bGain] = findBestSplit(X,y)
    [n,d]  = size(X);
    pG     = giniImpurity(y);
    bGain  = -inf; bFeat = 1; bThresh = 0;
    fSub   = randperm(d, min(d,round(sqrt(d))));
    for fi = fSub
        vals = unique(X(:,fi));
        if numel(vals)>20, vals = vals(round(linspace(1,numel(vals),20))); end
        for ti = 1:numel(vals)-1
            thr = (vals(ti)+vals(ti+1))/2;
            lY  = y(X(:,fi)<=thr); rY = y(X(:,fi)>thr);
            nL  = numel(lY); nR = numel(rY);
            if nL==0 || nR==0, continue; end
            wG = (nL*giniImpurity(lY) + nR*giniImpurity(rY)) / n;
            g  = pG - wG;
            if g>bGain, bGain=g; bFeat=fi; bThresh=thr; end
        end
    end
end

function node = buildTree(X,y,depth,maxDepth)
    node.isLeaf = false;
    node.label  = round(sum(y)/numel(y));
    node.feat   = 0; node.thresh = 0; node.left = []; node.right = [];
    if depth>=maxDepth || numel(unique(y))==1 || numel(y)<5
        node.isLeaf = true; return;
    end
    [feat,thresh,gain] = findBestSplit(X,y);
    if gain<=0, node.isLeaf = true; return; end
    lM = X(:,feat)<=thresh; rM = ~lM;
    if sum(lM)<2 || sum(rM)<2
        node.isLeaf = true; return;
    end
    node.feat = feat; node.thresh = thresh;
    node.left  = buildTree(X(lM,:), y(lM), depth+1, maxDepth);
    node.right = buildTree(X(rM,:), y(rM), depth+1, maxDepth);
end

function pred = predictTree(node, X)
    n = size(X,1); pred = zeros(n,1);
    for i = 1:n
        curr = node;
        while ~curr.isLeaf
            if X(i,curr.feat)<=curr.thresh
                curr = curr.left;
            else
                curr = curr.right;
            end
        end
        pred(i) = curr.label;
    end
end

% --- Random Forest helpers -------------------------------------------
function pred = predictForest(forest, X)
    nTrees = numel(forest); nTest = size(X,1);
    votes  = zeros(nTest,1);
    for t = 1:nTrees
        votes = votes + predictTree(forest{t}, X);
    end
    pred = double(votes/nTrees >= 0.5);
end

function scores = predictForestProb(forest, X)
    nTrees = numel(forest); nTest = size(X,1);
    totalVotes = zeros(nTest,1);
    for t = 1:nTrees
        totalVotes = totalVotes + predictTree(forest{t}, X);
    end
    scores = totalVotes / nTrees;
end

% --- Evaluation metrics ----------------------------------------------
function [acc,sens,spec,f1,auc] = evalMetrics(yTrue,yPred,scores)
    TP = sum(yTrue==1 & yPred==1); TN = sum(yTrue==0 & yPred==0);
    FP = sum(yTrue==0 & yPred==1); FN = sum(yTrue==1 & yPred==0);
    acc  = (TP+TN)/numel(yTrue);
    sens = TP/(TP+FN+1e-9);
    spec = TN/(TN+FP+1e-9);
    prec = TP/(TP+FP+1e-9);
    f1   = 2*prec*sens/(prec+sens+1e-9);
    if nargin>2 && ~isempty(scores)
        thr = sort(unique(scores),'descend');
        tprs = zeros(numel(thr)+2,1); fprs = zeros(numel(thr)+2,1);
        for ti = 1:numel(thr)
            pT  = double(scores>=thr(ti));
            tp_ = sum(yTrue==1 & pT==1); fp_ = sum(yTrue==0 & pT==1);
            fn_ = sum(yTrue==1 & pT==0); tn_ = sum(yTrue==0 & pT==0);
            tprs(ti+1) = tp_/(tp_+fn_+1e-9);
            fprs(ti+1) = fp_/(fp_+tn_+1e-9);
        end
        tprs(end) = 1; fprs(end) = 1;
        auc = 0;
        for ai = 1:numel(fprs)-1
            auc = auc + 0.5*(fprs(ai+1)-fprs(ai))*(tprs(ai)+tprs(ai+1));
        end
        auc = abs(auc);
    else
        auc = 0.5;
    end
end

% --- Confusion matrix ------------------------------------------------
function CM = confusionMat(yTrue,yPred)
    CM = zeros(2,2);
    CM(1,1) = sum(yTrue==0 & yPred==0);
    CM(1,2) = sum(yTrue==0 & yPred==1);
    CM(2,1) = sum(yTrue==1 & yPred==0);
    CM(2,2) = sum(yTrue==1 & yPred==1);
end

% --- Index shuffle helper --------------------------------------------
function out = shuffleIdx(idx)
    n = numel(idx);
    out = idx(randperm(n));
end