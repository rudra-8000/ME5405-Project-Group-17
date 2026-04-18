%[text] Labeling Separate Objects - Image Segmentation
%[text] Label the different objects using connected component labelling two-pass algorithm.
%[text] Referenced from p41-46 "Connected Components Operators, Chapter 3 - Binary Image and Color Image Processing, ME5405 Lecture Notes" and [https://en.wikipedia.org/wiki/Connected-component\_labeling\#Two-pass](https://en.wikipedia.org/wiki/Connected-component_labeling#Two-pass) and [https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman\_algorithm](https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman_algorithm) .
function [label_matrix, num_components] = ccl_segment(T, foreground)
% CCL_SEGMENT  Two-Pass Connected-Component Labeling
%
% Usage:
%   [label_matrix, num_components] = ccl_segment(T, foreground)
%
% Inputs:
%   T          — binary image matrix (values 0 and 1)
%   foreground — value that represents the foreground objects to be labeled
%                0 : foreground is black, background is white
%                1 : foreground is white, background is black
%
% Outputs:
%   label_matrix   — integer matrix, each unique positive integer
%                    corresponds to one connected component
%                    0 everywhere means background (no component)
%   num_components — total number of connected components found
%
% Connectivity: 8-connected (NW, N, NE, W neighbors checked)
%
% Example:
%   [labels, n] = ccl_segment(T, 0)  % black foreground
%   [labels, n] = ccl_segment(T, 1)  % white foreground

% ----------------------------------------------------------
% Normalise to internal convention: foreground = 0 (black)
% The algorithm checks (img == 0) to identify foreground pixels
% If input uses white foreground, flip before processing
% ----------------------------------------------------------
if foreground == 1
    img = 1 - T;
else
    img = double(T);
end

img          = double(img);
[rows, cols] = size(img);

fprintf('Segmenting image and labeling objects...\n');

% Label matrix initialised to 0 (background — no label assigned)
label_matrix = zeros(rows, cols);
max_labels   = rows * cols;
parent       = 1:max_labels;   % each label is its own root initially
NextLabel    = 0;

% ==========================================================
% PASS 1 — Provisional labeling and equivalence recording
% Scan left to right, top to bottom (raster order)
% 8-connectivity: check NW, N, NE, W neighbors
% ==========================================================

for r = 1:rows
    for c = 1:cols

        if img(r, c) == 0   % foreground pixel (black in internal convention)

            % --- Collect labels of visited neighbors ---
            neighbor_labels = [];

            if c > 1 && label_matrix(r, c-1) > 0                   % West
                neighbor_labels(end+1) = label_matrix(r, c-1);
            end
            if r > 1 && c > 1 && label_matrix(r-1, c-1) > 0        % North-West
                neighbor_labels(end+1) = label_matrix(r-1, c-1);
            end
            if r > 1 && label_matrix(r-1, c) > 0                    % North
                neighbor_labels(end+1) = label_matrix(r-1, c);
            end
            if r > 1 && c < cols && label_matrix(r-1, c+1) > 0     % North-East
                neighbor_labels(end+1) = label_matrix(r-1, c+1);
            end

            if isempty(neighbor_labels)
                % No labeled neighbors — create new label
                NextLabel         = NextLabel + 1;
                parent(NextLabel) = NextLabel;
                label_matrix(r, c) = NextLabel;

            else
                % Resolve each neighbor to its root, find minimum
                min_label = inf;
                for k = 1:length(neighbor_labels)
                    root = neighbor_labels(k);
                    while parent(root) ~= root
                        root = parent(root);
                    end
                    if root < min_label
                        min_label = root;
                    end
                end

                % Assign minimum root label to current pixel
                label_matrix(r, c) = min_label;

                % Union all neighbor roots to min_label
                for k = 1:length(neighbor_labels)
                    root = neighbor_labels(k);
                    while parent(root) ~= root
                        root = parent(root);
                    end
                    if root ~= min_label
                        parent(root) = min_label;
                    end
                end

            end

        end

    end
end

fprintf('Pass 1 complete. Provisional labels assigned: %d\n', NextLabel);

% ==========================================================
% BETWEEN PASSES — Resolve equivalences, renumber labels
% ==========================================================

% Fully flatten all chains in parent table to their root
for L = 1:NextLabel
    root = L;
    while parent(root) ~= root
        root = parent(root);
    end
    parent(L) = root;
end

% Assign sequential component numbers to each unique root
remap          = zeros(1, NextLabel);
num_components = 0;
for L = 1:NextLabel
    if parent(L) == L
        num_components = num_components + 1;
        remap(L)       = num_components;
    end
end

% Build final label lookup
final_label = zeros(1, NextLabel);
for L = 1:NextLabel
    final_label(L) = remap(parent(L));
end

fprintf('Equivalences resolved. Components found: %d\n', num_components);

% ==========================================================
% PASS 2 — Replace provisional labels with final labels
% ==========================================================

for r = 1:rows
    for c = 1:cols
        if label_matrix(r, c) > 0
            label_matrix(r, c) = final_label(label_matrix(r, c));
        end
    end
end

fprintf('Pass 2 complete. Labeling finished.\n');
fprintf('Total connected components: %d\n', num_components);

end

%[appendix]{"version":"1.0"}
%---
