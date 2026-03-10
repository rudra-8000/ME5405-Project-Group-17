function A = read_txt_image(filename)
% READ_TXT_IMAGE  Read a 64x64 image from a text file with 32 gray levels.
%   A = read_txt_image(filename)
%
%   The text file contains 64 lines of 64 alphanumeric characters each.
%   Characters '0'-'9' map to gray levels 0-9.
%   Characters 'A'-'V' map to gray levels 10-31.
%
%   Returns A as a 64x64 uint8 matrix.

    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    lf = char(10);   % line feed character
    cr = char(13);   % carriage return character
    
    % Read all characters, skipping CR and LF, into a 64x64 matrix
    A = fscanf(fid, [cr lf '%c'], [64, 64]);
    fclose(fid);
    
    % Transpose since fscanf fills column-by-column
    A = A';
    
    % Convert letter characters A-V to values 10-31
    A(isletter(A)) = A(isletter(A)) - 55;
    
    % Convert digit characters '0'-'9' to values 0-9
    A(A >= '0' & A <= '9') = A(A >= '0' & A <= '9') - 48;
    
    % Convert to uint8
    A = uint8(A);
end
