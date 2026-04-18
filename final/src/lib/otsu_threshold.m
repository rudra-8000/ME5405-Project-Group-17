%[text] Otsu's Method - Automatic Image Thresholding
%[text] Threshold the image and convert it into binary image (using Otsu's Method)
%[text] Referenced from page 747-751 Chapter 10.3 "Digital Image Processing 4e - R. C. Gonzales, R. E. Woods" and [https://en.wikipedia.org/wiki/Otsu%27s\_method](https://en.wikipedia.org/wiki/Otsu%27s_method)
function [T, threshold] = otsu_threshold(coded_array)
% OTSU_THRESHOLD  Automatic image thresholding using Otsu's Method
%
% Usage:
%   [T, threshold] = otsu_threshold(coded_array)
%
% Input:
%   coded_array — grayscale image matrix with intensity levels 0..31
%                 0 = darkest (black), 31 = brightest (white)
%
% Outputs:
%   T         — binary image matrix (0 = dark/foreground, 1 = bright/background)
%               pixels with intensity <= threshold → 0 (black)
%               pixels with intensity >  threshold → 1 (white)
%   threshold — optimal threshold level k* found by Otsu's method
%
% Referenced:
%   Gonzalez & Woods, Digital Image Processing 4e
%   Chapter 10.3, pp. 747-751
%   and Wikipedia: Otsu's method
%
% Example:
%   coded_array        = load_coded_image('chromo.txt');
%   [T, threshold]     = otsu_threshold(coded_array);
%   [T2, threshold2]   = otsu_threshold(coded_array2);

L = 32;
[rows, cols] = size(coded_array);
N_total      = rows * cols;

% ----------------------------------------------------------
% Step 1 — Count pixels at each intensity level
% ----------------------------------------------------------
counts = zeros(1, L);
for r = 1:rows
    for c = 1:cols
        k          = coded_array(r, c) + 1;
        counts(k)  = counts(k) + 1;
    end
end

% ----------------------------------------------------------
% Step 2 — Normalised histogram — probability of each level
% ----------------------------------------------------------
p = counts / N_total;

% ----------------------------------------------------------
% Step 3 — Cumulative probability P1(k)
% P1(k) = probability of intensity <= k
% ----------------------------------------------------------
p1 = zeros(1, L);
for k = 1:L
    for i = 1:k
        p1(k) = p1(k) + p(i);
    end
end

% ----------------------------------------------------------
% Step 4 — Cumulative mean mk
% mk(k) = sum of (intensity * probability) for levels 1..k
% ----------------------------------------------------------
mk = zeros(1, L);
for k = 1:L
    for i = 1:k
        mk(k) = mk(k) + (i * p(i));
    end
end

% ----------------------------------------------------------
% Step 5 — Global mean mg
% ----------------------------------------------------------
mg = mk(L);

% ----------------------------------------------------------
% Step 6 — Global variance var_g
% ----------------------------------------------------------
var_g = 0;
for r = 1:rows
    for c = 1:cols
        var_g = var_g + (coded_array(r, c) - mg)^2;
    end
end
var_g = var_g / N_total;

% ----------------------------------------------------------
% Step 7 — Between-class variance var_b for each threshold k
% Guard against division by zero at extremes (p1 = 0 or 1)
% ----------------------------------------------------------
var_b = zeros(1, L);
for k = 1:L
    denom = p1(k) * (1 - p1(k));
    if denom > 0
        var_b(k) = (mg * p1(k) - mk(k))^2 / denom;
    else
        var_b(k) = 0;
    end
end

% ----------------------------------------------------------
% Step 8 — Threshold effectiveness measure eta
% ----------------------------------------------------------
eff_thres = var_b / var_g;

% ----------------------------------------------------------
% Step 9 — Optimal threshold k* — maximum between-class variance
% ----------------------------------------------------------
[max_var_b, threshold] = max(var_b);

fprintf('Otsu threshold k* = %d | max var_b = %.4f | eta = %.4f\n', ...
         (threshold - 1), max_var_b, eff_thres(threshold));

% ----------------------------------------------------------
% Step 10 — Apply threshold to produce binary image T
% Pixels > threshold → white (1) background
% Pixels <= threshold → black (0) foreground
% ----------------------------------------------------------
T = zeros(rows, cols);
for r = 1:rows
    for c = 1:cols
        if coded_array(r, c) > (threshold - 1)
            T(r, c) = 1;
        else
            T(r, c) = 0;
        end
    end
end

end

%[appendix]{"version":"1.0"}
%---
