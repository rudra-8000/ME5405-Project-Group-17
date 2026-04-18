%[text] Determine Image Outline - Boundary Detection
%[text] Thresholding based on the boundary to determine the outline.
%[text] The method proceeds by scanning the given image 𝑓(𝑥,𝑦) of size 𝑁 by 𝑁.  A change in gray level from one band to the other denotes the presence of boundaries.
%[text] Pass 1 and 2 are to detect changes in the 𝑦 and 𝑥 directions, respectively. The combination of the results of the two passes will yield the boundary of the objects in the image.
%[text] Referenced from p29-33 "Thresholding based on the boundary, Chapter 3 - Binary Image and Color Image Processing, ME5405 Lecture Notes"
function bound = boundary_detect(T, foreground)
% BOUNDARY_DETECT  Two-pass row-column boundary detection
%
% Usage:
%   bound = boundary_detect(T, foreground)
%
% Inputs:
%   T          — binary image matrix (values 0 and 1)
%   foreground — value that represents the foreground object
%                0 : foreground is black, background is white
%                1 : foreground is white, background is black
%
% Output:
%   bound      — boundary image, same convention as input T
%                edge pixels carry the foreground value
%                non-edge pixels carry the background value
%
% Example:
%   bound = boundary_detect(T, 0)  % black foreground
%   bound = boundary_detect(T, 1)  % white foreground

% ----------------------------------------------------------
% Normalise to internal convention: foreground = 0 (black)
% LE = 0 (edge), LB = 1 (background)
% If input uses white foreground, flip before processing
% ----------------------------------------------------------
if foreground == 1
    img = 1 - T;
else
    img = double(T);
end

img          = double(img);
[rows, cols] = size(img);

LE = 0;   % edge pixel value
LB = 1;   % background pixel value

% Initialise intermediate images to LB (all background)
% First row and column have no prior neighbor — correctly stays LB
bound_inter1 = ones(rows, cols);   % g1 — horizontal transitions (Pass 1)
bound_inter2 = ones(rows, cols);   % g2 — vertical transitions   (Pass 2)

% ----------------------------------------------------------
% Pass 1 — scan each row, detect transitions in y direction
% Compare f(r,c) with left neighbor f(r, c-1)
% g1(r,c) = LE if different bands, LB otherwise
% ----------------------------------------------------------
for r = 2:rows
    for c = 2:cols
        if img(r,c) ~= img(r, c-1)
            bound_inter1(r,c) = LE;
        else
            bound_inter1(r,c) = LB;
        end
    end
end

% ----------------------------------------------------------
% Pass 2 — scan each column, detect transitions in x direction
% Compare f(r,c) with upper neighbor f(r-1, c)
% g2(r,c) = LE if different bands, LB otherwise
% ----------------------------------------------------------
for r = 2:rows
    for c = 2:cols
        if img(r,c) ~= img(r-1, c)
            bound_inter2(r,c) = LE;
        else
            bound_inter2(r,c) = LB;
        end
    end
end

% ----------------------------------------------------------
% Combination — equation (9)
% g(r,c) = LE if g1(r,c) = LE OR g2(r,c) = LE
%         = LB otherwise
% ----------------------------------------------------------
bound_img = ones(rows, cols);   % initialise to all background

for r = 1:rows
    for c = 1:cols
        if bound_inter1(r,c) == LE || bound_inter2(r,c) == LE
            bound_img(r,c) = LE;
        else
            bound_img(r,c) = LB;
        end
    end
end

% ----------------------------------------------------------
% Restore original convention
% If input had white foreground, flip result back
% ----------------------------------------------------------
if foreground == 1
    bound = 1 - bound_img;
else
    bound = bound_img;
end

end

%[appendix]{"version":"1.0"}
%---
