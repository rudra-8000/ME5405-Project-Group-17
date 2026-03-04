function [labeled, num_objects] = my_bwlabel(BW)
% MY_BWLABEL  Connected component labeling using flood fill (8-connectivity).
%   [labeled, num_objects] = my_bwlabel(BW)
%
%   BW - binary image (logical)
%   labeled - matrix of same size with integer labels (0 = background)
%   num_objects - number of connected components found

    [rows, cols] = size(BW);
    labeled = zeros(rows, cols);
    current_label = 0;
    
    % 8-connectivity neighbor offsets
    offsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    
    for r = 1:rows
        for c = 1:cols
            % If this pixel is foreground and not yet labeled
            if BW(r, c) && labeled(r, c) == 0
                current_label = current_label + 1;
                
                % BFS flood fill
                queue = [r, c];
                labeled(r, c) = current_label;
                
                while ~isempty(queue)
                    % Dequeue
                    cr = queue(1, 1);
                    cc = queue(1, 2);
                    queue(1, :) = [];
                    
                    % Check all 8 neighbors
                    for k = 1:8
                        nr = cr + offsets(k, 1);
                        nc = cc + offsets(k, 2);
                        
                        % Bounds check
                        if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                            if BW(nr, nc) && labeled(nr, nc) == 0
                                labeled(nr, nc) = current_label;
                                queue = [queue; nr, nc]; %#ok<AGROW>
                            end
                        end
                    end
                end
            end
        end
    end
    
    num_objects = current_label;
    fprintf('Found %d connected components\n', num_objects);
end
