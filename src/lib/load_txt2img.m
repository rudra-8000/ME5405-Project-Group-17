%[text] Reads a text file containing a coded array of an image of size 64 x 64 and 32 intensity levels.
%[text] Each pixel is represented by an alphanumeric character ranging from 0-9 and A-V corresponding to 32 levels of gray.
function coded_array = load_txt2img(filename)

% LOAD_CODED_IMAGE  Loads a coded image from a .txt file
%
% Usage:
%   coded_array = load_coded_image(filename)
%
% Input:
%   filename   — string, path to the .txt file containing the coded image
%                e.g. 'chromo.txt' or 'image2.txt'
%
% Output:
%   coded_array — 64x64 double matrix, intensity values 1..32
%                 1 = darkest (black), 32 = brightest (white)
%
% File format expected:
%   64x64 alphanumeric characters, one character per pixel
%   '0'-'9' map to intensity levels 1-10
%   'A'-'V' map to intensity levels 11-32
%   File is assumed to be stored row by row (standard image order)

% ----------------------------------------------------------
% Open file
% ----------------------------------------------------------
file_id = fopen(filename, 'r');
if file_id == -1
    error('Could not open file: %s — check the file path and name.', filename);
end

% ----------------------------------------------------------
% Read characters into a 64x64 matrix
% fscanf fills column by column
% ----------------------------------------------------------
format_spec = '%s';
size_array  = [64, 64];
coded_array = fscanf(file_id, format_spec, size_array);

% ----------------------------------------------------------
% Close file
% ----------------------------------------------------------
fclose(file_id);

% ----------------------------------------------------------
% Map alphanumeric characters to intensity levels 1..32
% '0'-'9' (10 chars) + 'A'-'V' (22 chars) = 32 levels total
% map is indexed by ASCII value of each character
% ----------------------------------------------------------
gray_levels = '0123456789ABCDEFGHIJKLMNOPQRSTUV';
map         = zeros(1, 32);
map(gray_levels) = 1:length(gray_levels);

coded_array = map(coded_array);

% ----------------------------------------------------------
% Validate — catch any unmapped or unexpected characters
% ----------------------------------------------------------
if any(coded_array(:) == 0)
    warning('coded_array contains unmapped values (0). Check "%s" for unexpected whitespace or invalid characters.', filename);
end

% ----------------------------------------------------------
% Transpose to correct row/column orientation
% fscanf fills column-major — transpose restores row-major layout
% ----------------------------------------------------------
coded_array = coded_array'- 1;

fprintf('Loaded "%s" — size: %dx%d — intensity range: [%d, %d]\n', ...
    filename, size(coded_array,1), size(coded_array,2), ...
    min(coded_array(:)), max(coded_array(:)));

end

%[appendix]{"version":"1.0"}
%---
