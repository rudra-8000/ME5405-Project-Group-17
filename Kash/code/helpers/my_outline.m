function outline = my_outline(BW)
% MY_OUTLINE  Extract one-pixel-thick outline from a binary image.
%   outline = my_outline(BW)
%
%   A foreground pixel is an outline pixel if at least one of its
%   4-connected neighbors is a background pixel (or is on the image border).

    [rows, cols] = size(BW);
    outline = false(rows, cols);
    
    % 4-connectivity neighbor offsets
    offsets = [-1 0; 1 0; 0 -1; 0 1];
    
    for r = 1:rows
        for c = 1:cols
            if BW(r, c)
                % Check if any 4-neighbor is background or out-of-bounds
                is_boundary = false;
                for k = 1:4
                    nr = r + offsets(k, 1);
                    nc = c + offsets(k, 2);
                    
                    if nr < 1 || nr > rows || nc < 1 || nc > cols
                        % Border pixel -> outline
                        is_boundary = true;
                        break;
                    elseif ~BW(nr, nc)
                        % Neighbor is background -> outline
                        is_boundary = true;
                        break;
                    end
                end
                
                if is_boundary
                    outline(r, c) = true;
                end
            end
        end
    end
end
