function [threshold, BW] = otsu_threshold(img, mode)
% OTSU_THRESHOLD  Compute optimal threshold using Otsu's method.
%   [threshold, BW] = otsu_threshold(img)
%   [threshold, BW] = otsu_threshold(img, mode)
%
%   img  - 2D grayscale image (uint8)
%   mode - 'dark_fg' (foreground is darker, default) or 'bright_fg'
%
%   Returns the optimal threshold and the binary image.
%   In 'dark_fg' mode: BW = (img < threshold)   -> chromosomes
%   In 'bright_fg' mode: BW = (img > threshold)  -> characters

    if nargin < 2
        mode = 'dark_fg';
    end
    
    img = double(img);
    min_val = min(img(:));
    max_val = max(img(:));
    num_pixels = numel(img);
    
    % Compute histogram
    num_levels = max_val - min_val + 1;
    levels = min_val:max_val;
    hist_counts = zeros(1, num_levels);
    for i = 1:num_levels
        hist_counts(i) = sum(img(:) == levels(i));
    end
    
    % Normalized histogram (probabilities)
    prob = hist_counts / num_pixels;
    
    % Try all possible thresholds and find the one that maximizes
    % the between-class variance
    best_sigma = 0;
    best_t = min_val;
    
    for t_idx = 1:(num_levels - 1)
        % Class 0: pixels with values <= levels(t_idx)
        % Class 1: pixels with values >  levels(t_idx)
        w0 = sum(prob(1:t_idx));          % weight of class 0
        w1 = sum(prob(t_idx+1:end));      % weight of class 1
        
        if w0 == 0 || w1 == 0
            continue;
        end
        
        % Mean of each class
        mu0 = sum(levels(1:t_idx) .* prob(1:t_idx)) / w0;
        mu1 = sum(levels(t_idx+1:end) .* prob(t_idx+1:end)) / w1;
        
        % Between-class variance
        sigma_b = w0 * w1 * (mu0 - mu1)^2;
        
        if sigma_b > best_sigma
            best_sigma = sigma_b;
            best_t = levels(t_idx);
        end
    end
    
    threshold = best_t;
    
    % Create binary image based on mode
    if strcmp(mode, 'bright_fg')
        BW = img > threshold;
    else
        BW = img < threshold;
    end
    
    BW = logical(BW);
    
    fprintf('Otsu threshold = %d (mode: %s)\n', threshold, mode);
end
