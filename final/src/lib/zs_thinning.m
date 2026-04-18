%[text] Zhang-Suen Thinning - Image Skeletonization
%[text] Determine a one-pixel thin image of the objects (using Zhang-Suen thinning algorithm to erode image). Reference: [https://rosettacode.org/wiki/Zhang-Suen\_thinning\_algorithm](https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm)
function thinned = zs_thinning(T, foreground)
% ZS_THINNING  Zhang-Suen thinning algorithm
%
% Usage:
%   thinned = zs_thinning(T, foreground)
%
% Inputs:
%   T          — binary image matrix (values 0 and 1)
%   foreground — value that represents the foreground object to be thinned
%                0 : foreground is black
%                1 : foreground is white
%
% Output:
%   thinned    — thinned binary image, same convention as input T
%                (foreground and background values are preserved as given)
%
% Example:
%   thinned = zs_thinning(T, 0)  % black foreground
%   thinned = zs_thinning(T, 1)  % white foreground

% ----------------------------------------------------------
% Normalise to algorithm convention: foreground = 0 (black)
% If the input uses white (1) as foreground, flip it first
% ----------------------------------------------------------
if foreground == 1
    zs_img = 1 - T;   % flip so foreground becomes 0
else
    zs_img = double(T);
end

fprintf('Creating One-Pixel Thin Image using Zhang-Suen Skeletonization...\n');

zs_img           = double(zs_img);
[rows, cols]  = size(zs_img);
changed       = true;
iter          = 0;

while changed

    iter = iter + 1;
    prev = zs_img;

    % ------------------------------------------------------
    % Pass 1 — South-East boundary removal
    % ------------------------------------------------------
    
    marker1 = false(rows, cols);

    for r = 2:rows-1
        for c = 2:cols-1
            
            % Neighbor layout (clockwise from top):
            %
            %   P9 | P2 | P3
            %   ───────────
            %   P8 | P1 | P4
            %   ───────────
            %   P7 | P6 | P5

            P1 = zs_img(r,   c  );
            P2 = zs_img(r-1, c  );
            P3 = zs_img(r-1, c+1);
            P4 = zs_img(r,   c+1);
            P5 = zs_img(r+1, c+1);
            P6 = zs_img(r+1, c  );
            P7 = zs_img(r+1, c-1);
            P8 = zs_img(r,   c-1);
            P9 = zs_img(r-1, c-1);

            % B(P): count black (0) neighbors — foreground
            B = (P2==0) + (P3==0) + (P4==0) + (P5==0) + ...
                (P6==0) + (P7==0) + (P8==0) + (P9==0);

            % A(P): count 1->0 transitions clockwise
            nb = [P2, P3, P4, P5, P6, P7, P8, P9, P2];
            A  = 0;
            for k = 1:8
                if nb(k) == 1 && nb(k+1) == 0
                    A = A + 1;
                end
            end

            if P1 == 0              && ...
               B >= 2 && B <= 6     && ...
               A == 1               && ...
               (P2 + P4 + P6) > 0   && ...
               (P4 + P6 + P8) > 0
                marker1(r, c) = true;
            end

        end
    end

    zs_img(marker1) = 1;   % delete marked pixels

    % ------------------------------------------------------
    % Pass 2 — North-West boundary removal
    % ------------------------------------------------------
    marker2 = false(rows, cols);

    for r = 2:rows-1
        for c = 2:cols-1

            P1 = zs_img(r,   c  );
            P2 = zs_img(r-1, c  );
            P3 = zs_img(r-1, c+1);
            P4 = zs_img(r,   c+1);
            P5 = zs_img(r+1, c+1);
            P6 = zs_img(r+1, c  );
            P7 = zs_img(r+1, c-1);
            P8 = zs_img(r,   c-1);
            P9 = zs_img(r-1, c-1);

            B = (P2==0) + (P3==0) + (P4==0) + (P5==0) + ...
                (P6==0) + (P7==0) + (P8==0) + (P9==0);

            nb = [P2, P3, P4, P5, P6, P7, P8, P9, P2];
            A  = 0;
            for k = 1:8
                if nb(k) == 1 && nb(k+1) == 0
                    A = A + 1;
                end
            end

            if P1 == 0              && ...
               B >= 2 && B <= 6     && ...
               A == 1               && ...
               (P2 + P4 + P8) > 0   && ...
               (P2 + P6 + P8) > 0
                marker2(r, c) = true;
            end

        end
    end

    zs_img(marker2) = 1;   % delete marked pixels

    changed = ~isequal(zs_img, prev);
    fprintf('Iteration %d | Pass1 removed: %d | Pass2 removed: %d\n', ...
             iter, sum(marker1(:)), sum(marker2(:)));

end

fprintf('Thinning complete in %d iterations.\n', iter);

% ----------------------------------------------------------
% Restore original convention
% If input had white foreground, flip result back
% ----------------------------------------------------------
if foreground == 1
    thinned = 1 - zs_img;
else
    thinned = zs_img;
end

end

%[appendix]{"version":"1.0"}
%---
