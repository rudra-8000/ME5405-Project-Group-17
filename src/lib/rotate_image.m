%[text] Image Rotation - Nearest-Neighbor Interpolation and Backwards Mapping
%[text] Rotate the output image from Step 6 about its center by *n* degrees.
%[text] Using nearest-neighbour interpolation and inverse/backwards mapping.
function rotated = rotate_image(T, angle_deg, direction, background)
% ROTATE_IMAGE  Rotates a binary image about its center
%               Output canvas is sized to contain the entire rotated image
%
% Uses backward mapping with nearest neighbor interpolation.
% Output is binary — same values (0 and 1) as input.
%
% Usage:
%   rotated = rotate_image(T, angle_deg)
%   rotated = rotate_image(T, angle_deg, direction)
%   rotated = rotate_image(T, angle_deg, direction, background)
%
% Inputs:
%   T          — binary image matrix (values 0 and 1)
%   angle_deg  — rotation angle in degrees (positive value)
%   direction  — (optional) rotation direction:
%                'cw'  : clockwise (default)
%                'ccw' : counter-clockwise
%   background — (optional) pixel value for out-of-bounds regions
%                1 : white background (default)
%                0 : black background
%
% Output:
%   rotated    — rotated binary image, expanded canvas size to fully
%                contain the rotated image with no clipping
%
% Example:
%   rotated = rotate_image(T, 30, 'cw', 1);
%   rotated = rotate_image(T, 30, 'ccw', 0);
%   rotated = rotate_image(T, 30);

% ----------------------------------------------------------
% Handle optional arguments
% ----------------------------------------------------------
if nargin < 3 || isempty(direction)
    direction = 'cw';
end
if nargin < 4 || isempty(background)
    background = 1;
end

% ----------------------------------------------------------
% Validate direction input
% ----------------------------------------------------------
if ~strcmp(direction, 'cw') && ~strcmp(direction, 'ccw')
    error('direction must be ''cw'' (clockwise) or ''ccw'' (counter-clockwise).');
end

% ----------------------------------------------------------
% Convert angle to radians with direction sign convention
% Negative angle → clockwise in image coordinates (y axis down)
% Positive angle → counter-clockwise in image coordinates
% ----------------------------------------------------------
if strcmp(direction, 'cw')
    theta = -deg2rad(angle_deg);
else
    theta =  deg2rad(angle_deg);
end

% ----------------------------------------------------------
% Input image dimensions and center
% ----------------------------------------------------------
[rows_in, cols_in] = size(T);
cx_in = (cols_in + 1) / 2;
cy_in = (rows_in + 1) / 2;

% ----------------------------------------------------------
% Compute output canvas size to contain entire rotated image
%
% Rotate all four corners of the input image and find the
% bounding box of those rotated corners. The canvas is sized
% to exactly fit this bounding box.
%
% The four corners in center-origin coordinates are:
%   top-left:     (-cx_in+0.5, -cy_in+0.5)
%   top-right:    ( cx_in-0.5, -cy_in+0.5)
%   bottom-left:  (-cx_in+0.5,  cy_in-0.5)
%   bottom-right: ( cx_in-0.5,  cy_in-0.5)
% ----------------------------------------------------------
cos_t = cos(theta);
sin_t = sin(theta);

% Corner offsets from center in center-origin coordinates
half_w = cols_in / 2;
half_h = rows_in / 2;

corners_x = [-half_w,  half_w, -half_w,  half_w];
corners_y = [-half_h, -half_h,  half_h,  half_h];

% Rotate each corner using forward rotation matrix
rotated_corners_x = cos_t * corners_x - sin_t * corners_y;
rotated_corners_y = sin_t * corners_x + cos_t * corners_y;

% Bounding box of rotated corners gives required canvas size
% ceil ensures the canvas fully contains all rotated content
cols_out = ceil(max(rotated_corners_x) - min(rotated_corners_x));
rows_out = ceil(max(rotated_corners_y) - min(rotated_corners_y));

% Center of the output canvas
cx_out = (cols_out + 1) / 2;
cy_out = (rows_out + 1) / 2;

fprintf('Input size:  %d x %d\n', rows_in, cols_in);
fprintf('Output size: %d x %d\n', rows_out, cols_out);

% ----------------------------------------------------------
% Inverse rotation components for backward mapping
% R_inv is rotation by -theta (transpose of forward R)
% ----------------------------------------------------------
cos_inv =  cos_t;    % cos(-theta) =  cos(theta)
sin_inv = -sin_t;    % sin(-theta) = -sin(theta)

% ----------------------------------------------------------
% Initialise output canvas to background color
% ----------------------------------------------------------
rotated = background * ones(rows_out, cols_out);

% ----------------------------------------------------------
% Backward mapping
%
% For each pixel (r,c) in the OUTPUT canvas:
%
%   Step 1: Translate from output canvas coordinates to
%           center-origin using OUTPUT center
%
%   Step 2: Apply inverse rotation to find source location
%           in center-origin coordinates
%
%   Step 3: Translate from center-origin back to INPUT
%           image coordinates using INPUT center
%
%   Step 4: Nearest neighbor — round to closest integer
%
%   Step 5: Bounds check — copy from input if inside bounds,
%           otherwise leave as background
% ----------------------------------------------------------
for r = 1:rows_out
    for c = 1:cols_out

        % Step 1 — translate to center-origin using output center
        xc = c - cx_out;
        yc = r - cy_out;

        % Step 2 — apply inverse rotation
        x_src_c =  cos_inv * xc + sin_inv * yc;
        y_src_c = -sin_inv * xc + cos_inv * yc;

        % Step 3 — translate back using INPUT image center
        x_src = x_src_c + cx_in;
        y_src = y_src_c + cy_in;

        % Step 4 — nearest neighbor interpolation
        c_src = round(x_src);
        r_src = round(y_src);

        % Step 5 — bounds check and pixel assignment
        if r_src >= 1 && r_src <= rows_in && ...
           c_src >= 1 && c_src <= cols_in
            rotated(r, c) = T(r_src, c_src);
        end

    end
end

fprintf('Rotation complete: %g degrees %s | background = %d\n', ...
    angle_deg, direction, background);

end

%[appendix]{"version":"1.0"}
%---
