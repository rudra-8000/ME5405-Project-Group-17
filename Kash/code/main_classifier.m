%% ME5405 Machine Vision - Image 2: SVM Classification (Tasks 2.8-2.9)
% This script trains an SVM classifier on the provided dataset,
% tests it on held-out data and on the segmented characters from Image 2.

close all; clear; clc;

addpath('helpers');
data_dir    = '../data/';
results_dir = '../results/';

%% Step 1: Load the dataset
fprintf('=== Step 1: Loading Dataset ===\n');

% Define the folder-to-label mapping
dataset_path = [data_dir 'p_dataset_26'];

folders = {'SampleA', 'SampleB', 'SampleC', 'Sample1', 'Sample2', 'Sample3'};
labels  = {'A',       'B',       'C',       '1',       '2',       '3'};

all_features = [];  % Each row is a flattened 26x26 = 676-element vector
all_labels = {};    % Cell array of character labels

for f = 1:length(folders)
    folder_path = fullfile(dataset_path, folders{f});
    mat_files = dir(fullfile(folder_path, '*.mat'));
    
    fprintf('  Loading %s (%s): %d files...', folders{f}, labels{f}, length(mat_files));
    
    for m = 1:length(mat_files)
        data = load(fullfile(folder_path, mat_files(m).name));
        img = double(data.imageArray);
        
        % CRITICAL: Invert polarity
        % Training images: white background (255), dark foreground
        % We invert so foreground is bright (matching Image 2 convention)
        img = 255 - img;
        
        % Flatten to feature vector
        feature_vec = img(:)';  % 1 x 676
        all_features = [all_features; feature_vec]; %#ok<AGROW>
        all_labels{end+1} = labels{f}; %#ok<SAGROW>
    end
    fprintf(' done\n');
end

all_labels = all_labels(:);  % Column cell array
fprintf('Total samples: %d\n', size(all_features, 1));
fprintf('Feature vector length: %d\n', size(all_features, 2));

%% Step 2: Split 75% train / 25% test
fprintf('\n=== Step 2: Train/Test Split (75/25) ===\n');

rng(42);  % For reproducibility

N = size(all_features, 1);
num_train = round(0.75 * N);

% Random permutation
perm = randperm(N);
train_idx = perm(1:num_train);
test_idx = perm(num_train+1:end);

X_train = all_features(train_idx, :);
Y_train = all_labels(train_idx);
X_test = all_features(test_idx, :);
Y_test = all_labels(test_idx);

fprintf('Training samples: %d\n', length(train_idx));
fprintf('Testing samples: %d\n', length(test_idx));

%% Step 3: Train baseline SVM (RBF kernel)
fprintf('\n=== Step 3: Train Baseline SVM Classifier ===\n');

% Use fitcecoc for multi-class SVM (one-vs-one strategy)
% Template with RBF kernel
t_svm = templateSVM('KernelFunction', 'rbf', ...
                     'BoxConstraint', 1, ...
                     'KernelScale', 'auto', ...
                     'Standardize', true);

fprintf('Training SVM with RBF kernel (auto kernel scale)...\n');
tic;
svm_model = fitcecoc(X_train, Y_train, 'Learners', t_svm);
train_time = toc;
fprintf('Training completed in %.1f seconds\n', train_time);

%% Step 4: Evaluate on held-out test set
fprintf('\n=== Step 4: Test Set Evaluation ===\n');

Y_pred = predict(svm_model, X_test);

% Compute accuracy
correct = sum(strcmp(Y_pred, Y_test));
accuracy = correct / length(Y_test) * 100;
fprintf('Test Accuracy: %.2f%% (%d/%d)\n', accuracy, correct, length(Y_test));

% Confusion matrix
fprintf('\nConfusion Matrix:\n');
unique_labels = {'A', 'B', 'C', '1', '2', '3'};
conf_mat = zeros(6, 6);
for i = 1:length(Y_test)
    true_idx = find(strcmp(unique_labels, Y_test{i}));
    pred_idx = find(strcmp(unique_labels, Y_pred{i}));
    conf_mat(true_idx, pred_idx) = conf_mat(true_idx, pred_idx) + 1;
end

% Print confusion matrix
fprintf('%8s', '');
for j = 1:6
    fprintf('%8s', ['Pred_' unique_labels{j}]);
end
fprintf('\n');
for i = 1:6
    fprintf('%8s', ['True_' unique_labels{i}]);
    for j = 1:6
        fprintf('%8d', conf_mat(i, j));
    end
    fprintf('\n');
end

% Plot confusion matrix
figure('Name', 'Confusion Matrix', 'NumberTitle', 'off');
imagesc(conf_mat);
colorbar;
set(gca, 'XTick', 1:6, 'XTickLabel', unique_labels);
set(gca, 'YTick', 1:6, 'YTickLabel', unique_labels);
xlabel('Predicted'); ylabel('True');
title(sprintf('Confusion Matrix (Accuracy: %.2f%%)', accuracy));
% Add numbers in cells
for i = 1:6
    for j = 1:6
        text(j, i, num2str(conf_mat(i,j)), ...
             'HorizontalAlignment', 'center', ...
             'FontWeight', 'bold', 'Color', 'w');
    end
end
saveas(gcf, [results_dir 'img2_08_confusion_matrix.png']);

%% Step 5: Classify characters from Image 2
fprintf('\n=== Step 5: Classify Image 2 Characters ===\n');

% Load segmented characters from main_image2.m
if ~exist([data_dir 'char_info.mat'], 'file')
    error('Please run main_image2.m first to generate char_info.mat');
end
load([data_dir 'char_info.mat'], 'char_info');

fprintf('\nClassification results for Image 2 characters:\n');
fprintf('%-10s %-15s %-15s %-10s\n', 'Object', 'True Label', 'Predicted', 'Correct?');
fprintf('%s\n', repmat('-', 1, 50));

num_chars = length(char_info);
img2_results = cell(num_chars, 3);

for k = 1:num_chars
    crop = double(char_info(k).cropped);
    
    % Center-pad to 26x26 to match training data dimensions
    % (center_pad preserves character shape better than resizing)
    [h, w] = size(crop);
    resized = zeros(26, 26);
    r_start = floor((26 - h) / 2) + 1;
    c_start = floor((26 - w) / 2) + 1;
    r_end = min(r_start + h - 1, 26);
    c_end = min(c_start + w - 1, 26);
    h_actual = r_end - r_start + 1;
    w_actual = c_end - c_start + 1;
    resized(r_start:r_end, c_start:c_end) = crop(1:h_actual, 1:w_actual);
    
    % Scale to 0-255 range (training data was inverted 0-255)
    resized = resized * 255;
    
    % Flatten to feature vector
    feature_vec = resized(:)';
    
    % Predict
    predicted = predict(svm_model, feature_vec);
    
    true_label = char_info(k).identity;
    is_correct = strcmp(predicted{1}, true_label);
    
    img2_results{k, 1} = true_label;
    img2_results{k, 2} = predicted{1};
    img2_results{k, 3} = is_correct;
    
    fprintf('%-10d %-15s %-15s %-10s\n', k, true_label, predicted{1}, ...
            ternary(is_correct, 'YES', 'NO'));
end

img2_correct = sum([img2_results{:, 3}]);
fprintf('\nImage 2 Classification: %d/%d correct\n', img2_correct, num_chars);

% Show the resized characters for visual verification
figure('Name', 'Image 2 - Resized for Classification', 'NumberTitle', 'off');
for k = 1:num_chars
    crop = double(char_info(k).cropped);
    [h, w] = size(crop);
    resized = zeros(26, 26);
    r_start = floor((26 - h) / 2) + 1;
    c_start = floor((26 - w) / 2) + 1;
    r_end = min(r_start + h - 1, 26);
    c_end = min(c_start + w - 1, 26);
    h_actual = r_end - r_start + 1;
    w_actual = c_end - c_start + 1;
    resized(r_start:r_end, c_start:c_end) = crop(1:h_actual, 1:w_actual) * 255;
    
    subplot(2, 3, k);
    imshow(uint8(resized), [0 255]);
    title(sprintf('True: %s, Pred: %s', img2_results{k,1}, img2_results{k,2}));
end
saveas(gcf, [results_dir 'img2_08_classification_results.png']);

%% Step 6: Hyperparameter Tuning & Preprocessing Experiments (Task 2.9)
fprintf('\n=== Step 6: Hyperparameter & Preprocessing Experiments ===\n');

% -------------------------------------------------------------------------
% Experiment 1: Kernel type comparison
% -------------------------------------------------------------------------
fprintf('\n--- Experiment 1: Kernel Type ---\n');
kernels = {'linear', 'rbf', 'polynomial'};
kernel_acc = zeros(1, length(kernels));

for ki = 1:length(kernels)
    fprintf('  Training with %s kernel...', kernels{ki});
    try
        t = templateSVM('KernelFunction', kernels{ki}, ...
                         'BoxConstraint', 1, ...
                         'KernelScale', 'auto', ...
                         'Standardize', true);
        mdl = fitcecoc(X_train, Y_train, 'Learners', t);
        pred = predict(mdl, X_test);
        kernel_acc(ki) = sum(strcmp(pred, Y_test)) / length(Y_test) * 100;
        fprintf(' Accuracy: %.2f%%\n', kernel_acc(ki));
    catch ME
        fprintf(' FAILED: %s\n', ME.message);
        kernel_acc(ki) = NaN;
    end
end

% -------------------------------------------------------------------------
% Experiment 2: Box Constraint (C) with RBF kernel
% -------------------------------------------------------------------------
fprintf('\n--- Experiment 2: Box Constraint (C) ---\n');
C_values = [0.1, 1, 10, 100];
C_acc = zeros(1, length(C_values));

for ci = 1:length(C_values)
    fprintf('  C = %.1f...', C_values(ci));
    t = templateSVM('KernelFunction', 'rbf', ...
                     'BoxConstraint', C_values(ci), ...
                     'KernelScale', 'auto', ...
                     'Standardize', true);
    mdl = fitcecoc(X_train, Y_train, 'Learners', t);
    pred = predict(mdl, X_test);
    C_acc(ci) = sum(strcmp(pred, Y_test)) / length(Y_test) * 100;
    fprintf(' Accuracy: %.2f%%\n', C_acc(ci));
end

% -------------------------------------------------------------------------
% Experiment 3: Kernel Scale (sigma) with RBF kernel
% -------------------------------------------------------------------------
fprintf('\n--- Experiment 3: Kernel Scale (RBF) ---\n');
sigma_values = [1, 5, 10, 50, 100];
sigma_acc = zeros(1, length(sigma_values));

for si = 1:length(sigma_values)
    fprintf('  sigma = %d...', sigma_values(si));
    t = templateSVM('KernelFunction', 'rbf', ...
                     'BoxConstraint', 1, ...
                     'KernelScale', sigma_values(si), ...
                     'Standardize', true);
    mdl = fitcecoc(X_train, Y_train, 'Learners', t);
    pred = predict(mdl, X_test);
    sigma_acc(si) = sum(strcmp(pred, Y_test)) / length(Y_test) * 100;
    fprintf(' Accuracy: %.2f%%\n', sigma_acc(si));
end

% -------------------------------------------------------------------------
% Experiment 4: Preprocessing methods for Image 2 characters
% -------------------------------------------------------------------------
fprintf('\n--- Experiment 4: Preprocessing Methods ---\n');
preprocess_methods = {'resize', 'center_pad', 'binarize_resize', 'resize_20x20_pad'};
preprocess_acc = zeros(1, length(preprocess_methods));

for pi = 1:length(preprocess_methods)
    method = preprocess_methods{pi};
    fprintf('  Method: %s\n', method);
    
    correct_count = 0;
    for k = 1:num_chars
        crop = double(char_info(k).cropped);
        
        switch method
            case 'resize'
                % Bilinear resize to 26x26
                processed = my_imresize(crop, [26, 26], 'bilinear') * 255;
                
            case 'center_pad'
                % Center-pad with zeros to 26x26
                [h, w] = size(crop);
                processed = zeros(26, 26);
                r_start = floor((26 - h) / 2) + 1;
                c_start = floor((26 - w) / 2) + 1;
                r_end = min(r_start + h - 1, 26);
                c_end = min(c_start + w - 1, 26);
                h_actual = r_end - r_start + 1;
                w_actual = c_end - c_start + 1;
                processed(r_start:r_end, c_start:c_end) = crop(1:h_actual, 1:w_actual) * 255;
                
            case 'binarize_resize'
                % Binarize first, then resize
                crop_bin = double(crop > 0);
                processed = my_imresize(crop_bin, [26, 26], 'nearest') * 255;
                
            case 'resize_20x20_pad'
                % Resize to 20x20, then center-pad to 26x26
                resized_small = my_imresize(crop, [20, 20], 'bilinear') * 255;
                processed = zeros(26, 26);
                processed(4:23, 4:23) = resized_small;
        end
        
        feature_vec = processed(:)';
        predicted = predict(svm_model, feature_vec);
        true_label = char_info(k).identity;
        
        if strcmp(predicted{1}, true_label)
            correct_count = correct_count + 1;
        end
        
        fprintf('    Char %s -> Predicted: %s\n', true_label, predicted{1});
    end
    
    preprocess_acc(pi) = correct_count / num_chars * 100;
    fprintf('    Method accuracy: %.1f%% (%d/%d)\n\n', ...
            preprocess_acc(pi), correct_count, num_chars);
end

%% Summary of all experiments
fprintf('\n========================================\n');
fprintf('SUMMARY OF EXPERIMENTS\n');
fprintf('========================================\n');

fprintf('\nExperiment 1 - Kernel Type (C=1, auto scale):\n');
for ki = 1:length(kernels)
    fprintf('  %-12s: %.2f%%\n', kernels{ki}, kernel_acc(ki));
end

fprintf('\nExperiment 2 - Box Constraint C (RBF, auto scale):\n');
for ci = 1:length(C_values)
    fprintf('  C = %-8.1f: %.2f%%\n', C_values(ci), C_acc(ci));
end

fprintf('\nExperiment 3 - Kernel Scale sigma (RBF, C=1):\n');
for si = 1:length(sigma_values)
    fprintf('  sigma = %-4d: %.2f%%\n', sigma_values(si), sigma_acc(si));
end

fprintf('\nExperiment 4 - Preprocessing (on Image 2 chars):\n');
for pi = 1:length(preprocess_methods)
    fprintf('  %-20s: %.1f%%\n', preprocess_methods{pi}, preprocess_acc(pi));
end

% Plot experiment results
figure('Name', 'Hyperparameter Experiments', 'NumberTitle', 'off');

subplot(2, 2, 1);
bar(kernel_acc);
set(gca, 'XTickLabel', kernels);
ylabel('Accuracy (%)');
title('Kernel Type Comparison');
ylim([0 100]);

subplot(2, 2, 2);
semilogx(C_values, C_acc, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Box Constraint C');
ylabel('Accuracy (%)');
title('Box Constraint vs Accuracy');
ylim([0 100]);

subplot(2, 2, 3);
semilogx(sigma_values, sigma_acc, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Kernel Scale \sigma');
ylabel('Accuracy (%)');
title('Kernel Scale vs Accuracy');
ylim([0 100]);

subplot(2, 2, 4);
bar(preprocess_acc);
set(gca, 'XTickLabel', {'resize', 'pad', 'bin+resize', 'resize+pad'});
ylabel('Accuracy (%)');
title('Preprocessing Method (Image 2)');
ylim([0 100]);

saveas(gcf, [results_dir 'img2_09_experiments.png']);

fprintf('\nAll experiments complete. Figures saved.\n');

%% Helper function
function result = ternary(cond, true_val, false_val)
    if cond
        result = true_val;
    else
        result = false_val;
    end
end