%% ME5405 Machine Vision - Image 1: Chromosomes (chromo.txt)
% This script performs all required image processing tasks on Image 1.
% The image shows chromosomes (dark objects on a bright background).

close all; clear; clc;

addpath('helpers');
data_dir    = '../data/';
results_dir = '../results/';

%% Task 1.1: Display the original image
fprintf('=== Task 1.1: Display Original Image ===\n');

img = read_txt_image([data_dir 'chromo.txt']);
fprintf('Image size: %d x %d\n', size(img, 1), size(img, 2));
fprintf('Gray level range: %d to %d\n', min(img(:)), max(img(:)));

figure('Name', 'Image 1 - Original', 'NumberTitle', 'off');
imshow(img, [0 31]);
title('Image 1: Original Chromosome Image (64x64, 32 gray levels)');
colorbar;
saveas(gcf, [results_dir 'img1_01_original.png']);

%% Task 1.2: Threshold into binary image
fprintf('\n=== Task 1.2: Binary Thresholding (Otsu''s Method) ===\n');

% Use custom Otsu's method
% Chromosomes are dark on bright background -> mode = 'dark_fg'
[thresh, BW] = otsu_threshold(img, 'dark_fg');

figure('Name', 'Image 1 - Binary', 'NumberTitle', 'off');
subplot(1, 3, 1);
imshow(img, [0 31]);
title('Original');

subplot(1, 3, 2);
imshow(BW);
title(sprintf('Binary (Otsu, T=%d)', thresh));

% Also show a manual threshold for comparison
manual_thresh = 15;
BW_manual = img < manual_thresh;
subplot(1, 3, 3);
imshow(BW_manual);
title(sprintf('Binary (Manual, T=%d)', manual_thresh));

saveas(gcf, [results_dir 'img1_02_binary.png']);

% Use the Otsu result for subsequent steps
fprintf('Using Otsu threshold = %d for subsequent processing\n', thresh);

%% Task 1.3: One-pixel thin image (Skeletonization)
fprintf('\n=== Task 1.3: Skeletonization ===\n');

% Custom Zhang-Suen thinning algorithm (no Toolbox dependency)
skeleton = my_thin(BW);

figure('Name', 'Image 1 - Skeleton', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(BW);
title('Binary Image');

subplot(1, 2, 2);
imshow(skeleton);
title('Skeletonized (One-pixel thin)');

saveas(gcf, [results_dir 'img1_03_skeleton.png']);

%% Task 1.4: Determine outlines
fprintf('\n=== Task 1.4: Outline Detection ===\n');

% Use custom outline function
outline = my_outline(BW);

figure('Name', 'Image 1 - Outline', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(BW);
title('Binary Image');

subplot(1, 2, 2);
imshow(outline);
title('Outlines of Chromosomes');

saveas(gcf, [results_dir 'img1_04_outline.png']);

%% Task 1.5: Label different objects
fprintf('\n=== Task 1.5: Connected Component Labeling ===\n');

% Use custom connected component labeling (8-connectivity)
[labeled, num_obj] = my_bwlabel(BW);

% Display with distinct colors per object
figure('Name', 'Image 1 - Labeled', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(BW);
title('Binary Image');

subplot(1, 2, 2);
% Create a color map for visualization
RGB_labeled = label2rgb(labeled, 'jet', 'k', 'shuffle');
imshow(RGB_labeled);
title(sprintf('Labeled Objects (%d found)', num_obj));

saveas(gcf, [results_dir 'img1_05_labeled.png']);

fprintf('\nImage 1 processing complete.\n');
fprintf('All figures saved as img1_*.png\n');
