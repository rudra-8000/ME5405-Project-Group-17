function thin = my_thin(BW)
% MY_THIN  Morphological thinning using the Zhang-Suen algorithm.
%   thin = my_thin(BW)
%
%   Produces a one-pixel-thin skeleton from a binary image.
%   BW  - input binary image (logical)
%   thin - thinned output (logical)

    thin = BW > 0;
    [rows, cols] = size(thin);
    
    changed = true;
    while changed
        changed = false;
        
        % --- Sub-iteration 1 ---
        to_remove = false(rows, cols);
        for r = 2:rows-1
            for c = 2:cols-1
                if ~thin(r, c), continue; end
                
                % 8-neighbors: P2(N) P3(NE) P4(E) P5(SE) P6(S) P7(SW) P8(W) P9(NW)
                P = [thin(r-1,c), thin(r-1,c+1), thin(r,c+1), thin(r+1,c+1), ...
                     thin(r+1,c), thin(r+1,c-1), thin(r,c-1), thin(r-1,c-1)];
                
                B = sum(P);  % number of non-zero neighbors
                if B < 2 || B > 6, continue; end
                
                % Count 0->1 transitions around the ring
                A = 0;
                for k = 1:8
                    next_k = mod(k, 8) + 1;
                    if P(k) == 0 && P(next_k) == 1
                        A = A + 1;
                    end
                end
                if A ~= 1, continue; end
                
                % Sub-iteration 1 conditions:
                % P2 * P4 * P6 == 0  AND  P4 * P6 * P8 == 0
                if P(1)*P(3)*P(5) == 0 && P(3)*P(5)*P(7) == 0
                    to_remove(r, c) = true;
                end
            end
        end
        if any(to_remove(:))
            thin(to_remove) = false;
            changed = true;
        end
        
        % --- Sub-iteration 2 ---
        to_remove = false(rows, cols);
        for r = 2:rows-1
            for c = 2:cols-1
                if ~thin(r, c), continue; end
                
                P = [thin(r-1,c), thin(r-1,c+1), thin(r,c+1), thin(r+1,c+1), ...
                     thin(r+1,c), thin(r+1,c-1), thin(r,c-1), thin(r-1,c-1)];
                
                B = sum(P);
                if B < 2 || B > 6, continue; end
                
                A = 0;
                for k = 1:8
                    next_k = mod(k, 8) + 1;
                    if P(k) == 0 && P(next_k) == 1
                        A = A + 1;
                    end
                end
                if A ~= 1, continue; end
                
                % Sub-iteration 2 conditions:
                % P2 * P4 * P8 == 0  AND  P2 * P6 * P8 == 0
                if P(1)*P(3)*P(7) == 0 && P(1)*P(5)*P(7) == 0
                    to_remove(r, c) = true;
                end
            end
        end
        if any(to_remove(:))
            thin(to_remove) = false;
            changed = true;
        end
    end
end
