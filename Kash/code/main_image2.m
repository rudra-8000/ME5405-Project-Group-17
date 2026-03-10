%% ME5405 Machine Vision - Image 2: Characters (charact1.txt)
% This script performs Tasks 2.1-2.7 on Image 2.
% The image shows characters 1,2,3 (top row) and A,B,C (bottom row)
% as bright objects on a black background.

close all; clear; clc;

addpath('helpers');
data_dir    = '../data/';
results_dir = '../results/';

%% Task 2.1: Display the original image
fprintf('=== Task 2.1: Display Original Image ===\n');

img = read_txt_image([data_dir 'charact1.txt']);
fprintf('Image size: %d x %d\n', size(img, 1), size(img, 2));
fprintf('Gray level range: %d to %d\n', min(img(:)), max(img(:)));

figure('Name', 'Image 2 - Original', 'NumberTitle', 'off');
imshow(img, [0 31]);
title('Image 2: Original Character Image (64x64, 32 gray levels)');
colorbar;
saveas(gcf, [results_dir 'img2_01_original.png']);

%% Task 2.2: Binary image using thresholding
fprintf('\n=== Task 2.2: Binary Thresholding (Otsu''s Method) ===\n');

% Characters are bright on black background -> mode = 'bright_fg'
[thresh, BW] = otsu_threshold(img, 'bright_fg');

figure('Name', 'Image 2 - Binary', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(img, [0 31]);
title('Original');

subplot(1, 2, 2);
imshow(BW);
title(sprintf('Binary (Otsu, T=%d)', thresh));

saveas(gcf, [results_dir 'img2_02_binary.png']);

%% Task 2.3: One-pixel thin image (Skeletonization)
fprintf('\n=== Task 2.3: Skeletonization ===\n');

% Custom Zhang-Suen thinning algorithm (no Toolbox dependency)
skeleton = my_thin(BW);

figure('Name', 'Image 2 - Skeleton', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(BW);
title('Binary Image');

subplot(1, 2, 2);
imshow(skeleton);
title('Skeletonized (One-pixel thin)');

saveas(gcf, [results_dir 'img2_03_skeleton.png']);

%% Task 2.4: Outline detection
fprintf('\n=== Task 2.4: Outline Detection ===\n');

outline = my_outline(BW);

figure('Name', 'Image 2 - Outline', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(BW);
title('Binary Image');

subplot(1, 2, 2);
imshow(outline);
title('Outlines of Characters');

saveas(gcf, [results_dir 'img2_04_outline.png']);

%% Task 2.5: Segment and label characters
fprintf('\n=== Task 2.5: Segment and Label Characters ===\n');

% Connected component labeling
[labeled, num_obj] = my_bwlabel(BW);
fprintf('Number of characters found: %d\n', num_obj);

% Compute centroid and bounding box for each component
char_info = struct('label', {}, 'centroid_r', {}, 'centroid_c', {}, ...
                   'r_min', {}, 'r_max', {}, 'c_min', {}, 'c_max', {}, ...
                   'cropped', {}, 'identity', {});

for k = 1:num_obj
    [r_coords, c_coords] = find(labeled == k);
    
    char_info(k).label = k;
    char_info(k).centroid_r = mean(r_coords);
    char_info(k).centroid_c = mean(c_coords);
    char_info(k).r_min = min(r_coords);
    char_info(k).r_max = max(r_coords);
    char_info(k).c_min = min(c_coords);
    char_info(k).c_max = max(c_coords);
    
    % Crop the binary character
    char_info(k).cropped = BW(char_info(k).r_min:char_info(k).r_max, ...
                               char_info(k).c_min:char_info(k).c_max);
    
    fprintf('  Object %d: centroid=(%.1f, %.1f), bbox=[%d-%d, %d-%d], size=%dx%d\n', ...
        k, char_info(k).centroid_r, char_info(k).centroid_c, ...
        char_info(k).r_min, char_info(k).r_max, ...
        char_info(k).c_min, char_info(k).c_max, ...
        char_info(k).r_max - char_info(k).r_min + 1, ...
        char_info(k).c_max - char_info(k).c_min + 1);
end

% Identify characters by spatial position
% Top row (small centroid_r) = 1, 2, 3 (left to right)
% Bottom row (large centroid_r) = A, B, C (left to right)
centroids_r = [char_info.centroid_r];
centroids_c = [char_info.centroid_c];

% Split into top and bottom rows using the midpoint of row centroids
mid_row = (min(centroids_r) + max(centroids_r)) / 2;

top_indices = find(centroids_r < mid_row);
bot_indices = find(centroids_r >= mid_row);

% Sort each row by column position (left to right)
[~, top_order] = sort(centroids_c(top_indices));
top_sorted = top_indices(top_order);

[~, bot_order] = sort(centroids_c(bot_indices));
bot_sorted = bot_indices(bot_order);

% Assign identities
% Top row left-to-right: '1', '2', '3'
char_names_top = {'1', '2', '3'};
for i = 1:length(top_sorted)
    char_info(top_sorted(i)).identity = char_names_top{i};
end

% Bottom row left-to-right: 'A', 'B', 'C'
char_names_bot = {'A', 'B', 'C'};
for i = 1:length(bot_sorted)
    char_info(bot_sorted(i)).identity = char_names_bot{i};
end

fprintf('\nCharacter identification:\n');
for k = 1:num_obj
    fprintf('  Object %d -> ''%s''\n', k, char_info(k).identity);
end

% Display labeled image with colors
figure('Name', 'Image 2 - Labeled', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(BW);
title('Binary Image');

subplot(1, 2, 2);
RGB_labeled = label2rgb(labeled, 'jet', 'k', 'shuffle');
imshow(RGB_labeled);
title(sprintf('Segmented Characters (%d found)', num_obj));

saveas(gcf, [results_dir 'img2_05_segmented.png']);

% Show individual cropped characters
figure('Name', 'Image 2 - Individual Characters', 'NumberTitle', 'off');
for k = 1:num_obj
    subplot(2, 3, k);
    imshow(char_info(k).cropped);
    title(sprintf('Character: %s', char_info(k).identity));
end
saveas(gcf, [results_dir 'img2_05b_individual_chars.png']);

%% Task 2.6: Arrange characters as AB123C
fprintf('\n=== Task 2.6: Arrange as AB123C ===\n');

% Desired order: A, B, 1, 2, 3, C
desired_order = {'A', 'B', '1', '2', '3', 'C'};

% Find the index in char_info for each desired character
ordered_indices = zeros(1, length(desired_order));
for i = 1:length(desired_order)
    for k = 1:num_obj
        if strcmp(char_info(k).identity, desired_order{i})
            ordered_indices(i) = k;
            break;
        end
    end
end

% Determine max height for vertical centering
max_h = 0;
for i = 1:length(ordered_indices)
    idx = ordered_indices(i);
    [h, ~] = size(char_info(idx).cropped);
    if h > max_h
        max_h = h;
    end
end

% Build the arranged image with spacing
spacing = 3;  % pixels between characters
total_width = 0;
for i = 1:length(ordered_indices)
    idx = ordered_indices(i);
    [~, w] = size(char_info(idx).cropped);
    total_width = total_width + w;
end
total_width = total_width + spacing * (length(ordered_indices) - 1);

arranged = false(max_h, total_width);

current_col = 1;
for i = 1:length(ordered_indices)
    idx = ordered_indices(i);
    crop = char_info(idx).cropped;
    [h, w] = size(crop);
    
    % Vertical centering
    row_offset = floor((max_h - h) / 2);
    
    arranged(row_offset+1:row_offset+h, current_col:current_col+w-1) = crop;
    current_col = current_col + w + spacing;
end

figure('Name', 'Image 2 - AB123C Arrangement', 'NumberTitle', 'off');
imshow(arranged);
title('Characters arranged as: A B 1 2 3 C');
saveas(gcf, [results_dir 'img2_06_AB123C.png']);

fprintf('Arranged image size: %d x %d\n', size(arranged, 1), size(arranged, 2));

%% Task 2.7: Rotate by 30 degrees about center
fprintf('\n=== Task 2.7: Rotate 30 degrees ===\n');

% Use custom rotation function
rotated = my_rotate(arranged, 30);

figure('Name', 'Image 2 - Rotated 30°', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(arranged);
title('Original AB123C');

subplot(1, 2, 2);
imshow(rotated);
title('Rotated 30° (custom implementation)');

saveas(gcf, [results_dir 'img2_07_rotated30.png']);

fprintf('Rotated image size: %d x %d\n', size(rotated, 1), size(rotated, 2));

%% Save workspace for classifier script
% Save char_info so the classifier script can use the segmented characters
save([data_dir 'char_info.mat'], 'char_info');
fprintf('\nSaved char_info.mat for use by the classifier.\n');

fprintf('\nImage 2 Tasks 2.1-2.7 complete.\n');
fprintf('All figures saved as img2_*.png\n');
