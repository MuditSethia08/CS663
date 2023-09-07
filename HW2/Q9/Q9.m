clear; close all;

% Load images
originalImage1 = imread("LC1.png");
originalImage2 = imread("LC2.jpg");

% Display original images
figure(1);
imagesc(originalImage1);
colormap('gray');
title("Original Image 1");

figure(2);
imagesc(originalImage2);
colormap('gray');
title("Original Image 2");

% Parameters
binWidth = 7;

% Local histogram equalization
localEqualizedImage1 = localHistogramEqualization(originalImage1, binWidth);
localEqualizedImage2 = localHistogramEqualization(originalImage2, binWidth);

% Display local histogram equalized images
figure(3);
imagesc(localEqualizedImage1);
colormap('gray');
title("Local Histogram Equalization of Image 1 with Bin Width = " + num2str(binWidth));

figure(4);
imagesc(localEqualizedImage2);
colormap('gray');
title("Local Histogram Equalization of Image 2 with Bin Width = " + num2str(binWidth));

% Global histogram equalization
globalEqualizedImage1 = globalHistogramEqualization(originalImage1);
globalEqualizedImage2 = globalHistogramEqualization(originalImage2);

% Display global histogram equalized images
figure(5);
imagesc(globalEqualizedImage1);
colormap('gray');
title("Global Histogram Equalization of Image 1");

figure(6);
imagesc(globalEqualizedImage2);
colormap('gray');
title("Global Histogram Equalization of Image 2");

%% Functions %%

% Local Histogram Equalization
function localEqualizedImage = localHistogramEqualization(inputImage, binWidth)
    localEqualizedImage = zeros(size(inputImage));
    [rows, cols] = size(inputImage);
    for x = 1:cols
        for y = 1:rows
            % Define local bin region
            startRow = max(y - ((binWidth - 1) / 2), 1);
            endRow = min(y + ((binWidth - 1) / 2), rows);
            startCol = max(x - ((binWidth - 1) / 2), 1);
            endCol = min(x + ((binWidth - 1) / 2), cols);
            
            % Extract the local region
            localRegion = inputImage(startRow:endRow, startCol:endCol);
            
            % Compute equalized pixel for the current pixel
            localEqualizedImage(y, x) = getEqualizedPixel(localRegion, (x - startCol) + 1, (y - startRow) + 1);
        end
    end
end

% Get Equalized Pixel for a Local Region
function equalizedPixel = getEqualizedPixel(localRegion, x, y)
    % Calculate pixel probability mass function (PMF) for the local region
    localRegion = localRegion + 1;
    pmf = accumarray(localRegion(:), ones(1, numel(localRegion)));
    pmf = [pmf; zeros(256 - numel(pmf), 1)];
    pmf = pmf / numel(localRegion);
    
    % Calculate cumulative distribution function (CDF)
    cdf = cumsum(pmf);
    
    % Calculate equalized pixel value
    equalizedPixel = round(255 * cdf(localRegion(y, x) + 1));
end

% Global Histogram Equalization
function globalEqualizedImage = globalHistogramEqualization(inputImage)
    globalEqualizedImage = histeq(inputImage, 256);
end
