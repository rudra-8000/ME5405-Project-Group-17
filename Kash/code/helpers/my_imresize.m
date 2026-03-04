function out = my_imresize(img, target_size, method)
% MY_IMRESIZE  Resize an image to target dimensions.
%   out = my_imresize(img, [rows, cols])
%   out = my_imresize(img, [rows, cols], method)
%
%   img         - input 2D image (any numeric or logical type)
%   target_size - [target_rows, target_cols]
%   method      - 'nearest' or 'bilinear' (default: 'bilinear')

    if nargin < 3
        method = 'bilinear';
    end
    
    img = double(img);
    [src_rows, src_cols] = size(img);
    tgt_rows = target_size(1);
    tgt_cols = target_size(2);
    
    out = zeros(tgt_rows, tgt_cols);
    
    % Scale factors
    row_scale = src_rows / tgt_rows;
    col_scale = src_cols / tgt_cols;
    
    for r = 1:tgt_rows
        for c = 1:tgt_cols
            % Map target pixel to source coordinates
            src_r = (r - 0.5) * row_scale + 0.5;
            src_c = (c - 0.5) * col_scale + 0.5;
            
            if strcmp(method, 'nearest')
                % Nearest neighbor
                nr = round(src_r);
                nc = round(src_c);
                nr = max(1, min(src_rows, nr));
                nc = max(1, min(src_cols, nc));
                out(r, c) = img(nr, nc);
                
            else
                % Bilinear interpolation
                r1 = floor(src_r);
                r2 = r1 + 1;
                c1 = floor(src_c);
                c2 = c1 + 1;
                
                % Clamp to image bounds
                r1 = max(1, min(src_rows, r1));
                r2 = max(1, min(src_rows, r2));
                c1 = max(1, min(src_cols, c1));
                c2 = max(1, min(src_cols, c2));
                
                % Fractional parts
                dr = src_r - floor(src_r);
                dc = src_c - floor(src_c);
                
                % Bilinear formula
                out(r, c) = (1-dr)*(1-dc)*img(r1,c1) + ...
                            (1-dr)*(dc)  *img(r1,c2) + ...
                            (dr)  *(1-dc)*img(r2,c1) + ...
                            (dr)  *(dc)  *img(r2,c2);
            end
        end
    end
end
