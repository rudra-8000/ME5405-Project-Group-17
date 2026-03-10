function rotated = my_rotate(img, angle_deg)
% MY_ROTATE  Rotate an image about its center by a given angle.
%   rotated = my_rotate(img, angle_deg)
%
%   Uses inverse mapping with nearest-neighbor interpolation.
%   The output image is expanded to fit the full rotated image.
%   angle_deg - rotation angle in degrees (positive = counter-clockwise)

    [rows, cols] = size(img);
    angle_rad = angle_deg * pi / 180;
    
    % Compute the size of the output image by rotating the 4 corners
    cx = (cols + 1) / 2;  % center x (column)
    cy = (rows + 1) / 2;  % center y (row)
    
    corners_x = [1, cols, 1, cols];
    corners_y = [1, 1, rows, rows];
    
    % Forward rotation of corners to find output bounds
    rot_x = cos(angle_rad) * (corners_x - cx) - sin(angle_rad) * (corners_y - cy) + cx;
    rot_y = sin(angle_rad) * (corners_x - cx) + cos(angle_rad) * (corners_y - cy) + cy;
    
    min_x = floor(min(rot_x));
    max_x = ceil(max(rot_x));
    min_y = floor(min(rot_y));
    max_y = ceil(max(rot_y));
    
    new_cols = max_x - min_x + 1;
    new_rows = max_y - min_y + 1;
    
    % New center in the expanded image
    new_cx = cx - min_x + 1;
    new_cy = cy - min_y + 1;
    
    % Initialize output
    if isa(img, 'logical')
        rotated = false(new_rows, new_cols);
    else
        rotated = zeros(new_rows, new_cols, class(img));
    end
    
    % Inverse mapping: for each output pixel, find the source pixel
    for r = 1:new_rows
        for c = 1:new_cols
            % Position relative to center in output
            dx = c - new_cx;
            dy = r - new_cy;
            
            % Inverse rotation (rotate by -angle)
            src_x = cos(-angle_rad) * dx - sin(-angle_rad) * dy + cx;
            src_y = sin(-angle_rad) * dx + cos(-angle_rad) * dy + cy;
            
            % Nearest-neighbor interpolation
            src_c = round(src_x);
            src_r = round(src_y);
            
            % Check bounds
            if src_r >= 1 && src_r <= rows && src_c >= 1 && src_c <= cols
                rotated(r, c) = img(src_r, src_c);
            end
        end
    end
end
