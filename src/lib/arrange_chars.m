%[text] Arranging Components - Characters Sequence
%[text] Arrange the characters in one line with the sequence.
function arranged = arrange_chars(T, label_matrix, component_sequence, background, spacing, padding)
% ARRANGE_CHARS  Extracts and arranges CCL components in a specified sequence
%
% Usage:
%   arranged = arrange_chars(T, label_matrix, component_sequence)
%   arranged = arrange_chars(T, label_matrix, component_sequence, background)
%   arranged = arrange_chars(T, label_matrix, component_sequence, background, spacing)
%   arranged = arrange_chars(T, label_matrix, component_sequence, background, spacing, padding)
%
% Inputs:
%   T                  — binary image (values 0 and 1)
%   label_matrix       — integer label matrix from ccl_segment
%                        0 = background, 1..n = component labels
%   component_sequence — row vector of component IDs in desired output order
%                        e.g. [1 4 5] produces components 1, 4, 5 left to right
%   background         — (optional) pixel value used for background, spacing,
%                        and padding. Should match image convention:
%                        1 : background is white (foreground is black) — default
%                        0 : background is black (foreground is white)
%   spacing            — (optional) number of background-colored pixels between
%                        characters. Default: 3
%   padding            — (optional) number of background-colored pixels added
%                        around each character crop on all four sides.
%                        Default: 2
%
% Output:
%   arranged           — binary image with selected components arranged
%                        in a single row in the specified sequence
%                        empty space filled with background color
%
% Example:
%   % Minimal call — all optional args use defaults
%   arranged = arrange_chars(T, label_matrix, [1 4 6]);
%
%   % Black background, white foreground, default spacing and padding
%   arranged = arrange_chars(T, label_matrix, [4 5 1 2 3 6], 0);
%
%   % White background, custom spacing
%   arranged = arrange_chars(T, label_matrix, [4 5 1 2 3 6], 1, 5);
%
%   % Full specification
%   arranged = arrange_chars(T, label_matrix, [4 5 1 2 3 6], 1, 5, 4);

% ----------------------------------------------------------
% Handle optional arguments
% ----------------------------------------------------------
if nargin < 4 || isempty(background)
    background = 1;   % default: white background, black foreground
end
if nargin < 5 || isempty(spacing)
    spacing = 3;
end
if nargin < 6 || isempty(padding)
    padding = 2;
end

num_selected = length(component_sequence);
[rows, cols] = size(label_matrix);

% ----------------------------------------------------------
% Step 1 — Extract bounding box for each selected component
% ----------------------------------------------------------
r_min = zeros(1, num_selected);
r_max = zeros(1, num_selected);
c_min = zeros(1, num_selected);
c_max = zeros(1, num_selected);

for idx = 1:num_selected

    k = component_sequence(idx);

    if ~any(label_matrix(:) == k)
        error('Component ID %d not found in label_matrix.', k);
    end

    r_min(idx) = rows;
    r_max(idx) = 1;
    c_min(idx) = cols;
    c_max(idx) = 1;

    for r = 1:rows
        for c = 1:cols
            if label_matrix(r, c) == k
                if r < r_min(idx), r_min(idx) = r; end
                if r > r_max(idx), r_max(idx) = r; end
                if c < c_min(idx), c_min(idx) = c; end
                if c > c_max(idx), c_max(idx) = c; end
            end
        end
    end

end

% ----------------------------------------------------------
% Step 2 — Compute padded bounding box extents
% Clamped to image boundaries to avoid out-of-bounds indexing
% ----------------------------------------------------------
r_min_pad = max(r_min - padding, 1);
r_max_pad = min(r_max + padding, rows);
c_min_pad = max(c_min - padding, 1);
c_max_pad = min(c_max + padding, cols);

crop_heights = r_max_pad - r_min_pad + 1;
crop_widths  = c_max_pad - c_min_pad + 1;

fprintf('Component bounding boxes (padding = %d):\n', padding);
for idx = 1:num_selected
    fprintf('  Component %d — rows %d:%d cols %d:%d — size %dx%d\n', ...
        component_sequence(idx), ...
        r_min_pad(idx), r_max_pad(idx), ...
        c_min_pad(idx), c_max_pad(idx), ...
        crop_heights(idx), crop_widths(idx));
end

% ----------------------------------------------------------
% Step 3 — Compute output canvas dimensions
% ----------------------------------------------------------
canvas_height = max(crop_heights);
canvas_width  = sum(crop_widths) + spacing * (num_selected - 1);

fprintf('Canvas: %d x %d | background = %d | spacing = %d | padding = %d\n', ...
    canvas_height, canvas_width, background, spacing, padding);

% ----------------------------------------------------------
% Step 4 — Initialise canvas to background color
% ----------------------------------------------------------
arranged = background * ones(canvas_height, canvas_width);

% ----------------------------------------------------------
% Step 5 — Paste each padded component crop onto canvas
% ----------------------------------------------------------
col_offset = 1;

for idx = 1:num_selected

    k = component_sequence(idx);

    % Crop from T using padded bounding box
    crop = T(r_min_pad(idx):r_max_pad(idx), ...
              c_min_pad(idx):c_max_pad(idx));

    % Centre vertically on canvas
    row_offset = floor((canvas_height - crop_heights(idx)) / 2) + 1;

    % Paste onto canvas
    arranged(row_offset : row_offset + crop_heights(idx) - 1, ...
             col_offset : col_offset + crop_widths(idx)  - 1) = crop;

    fprintf('  Placed component %d at rows %d:%d cols %d:%d\n', ...
        k, ...
        row_offset, row_offset + crop_heights(idx) - 1, ...
        col_offset, col_offset + crop_widths(idx)  - 1);

    col_offset = col_offset + crop_widths(idx) + spacing;

end

end

%[appendix]{"version":"1.0"}
%---
