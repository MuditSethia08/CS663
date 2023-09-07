clear; close all;

% Read the images
originalBarbara = im2double(imread('barbara256.png'));
originalKodak = im2double(imread('kodak24.png'));

% Display original images
figure(1); imagesc(originalBarbara); colormap("gray"); title("Original barbara256");
figure(2); imagesc(originalKodak); colormap("gray"); title("Original kodak24");

% Set noise and filter parameters
noiseStdDev = 5 / 255; % Noise standard deviation
spatialStdDev = 2;     % Spatial standard deviation
rangeStdDev = 2 / 255; % Range standard deviation (scaled for 0-1 range)

% Add noise to the images
noisyBarbara = originalBarbara + noiseStdDev * randn(size(originalBarbara));
noisyKodak = originalKodak + noiseStdDev * randn(size(originalKodak));

% Display noisy images
figure(3); imagesc(noisyBarbara); colormap("gray"); title("Noisy barbara256 with \sigma_n = " + num2str(noiseStdDev * 255));
figure(4); imagesc(noisyKodak); colormap("gray"); title("Noisy kodak24 with \sigma_n = " + num2str(noiseStdDev * 255));

% Apply bilateral filter
filteredBarbara = myBilateralFilter(noisyBarbara, spatialStdDev, rangeStdDev);
filteredKodak = myBilateralFilter(noisyKodak, spatialStdDev, rangeStdDev);

% Display filtered images
figure(5); imagesc(filteredBarbara); colormap("gray"); 
title("Filtered barbara256 with \sigma_n = " + num2str(noiseStdDev * 255) + ", \sigma_s = " + num2str(spatialStdDev) + ", \sigma_r = " + num2str(rangeStdDev * 255));
figure(6); imagesc(filteredKodak); colormap("gray");
title("Filtered kodak24 with \sigma_n = " + num2str(noiseStdDev * 255) + ", \sigma_s = " + num2str(spatialStdDev) + ", \sigma_r = " + num2str(rangeStdDev * 255));

% Define bilateral filter function
function filteredImage = myBilateralFilter(inputImage, spatialStdDev, rangeStdDev)
    [rows, cols] = size(inputImage);
    filteredImage = zeros(rows, cols);

    for x = 1:cols
        for y = 1:rows
            i1 = max(y - ceil(3 * spatialStdDev), 1);
            i2 = min(y + ceil(3 * spatialStdDev), rows);
            j1 = max(x - ceil(3 * spatialStdDev), 1);
            j2 = min(x + ceil(3 * spatialStdDev), cols);

            localPatch = inputImage(i1:i2, j1:j2);
            [X, Y] = meshgrid(j1:j2, i1:i2);

            spatialGaussian = exp(-((X - x).^2 + (Y - y).^2) / (2 * spatialStdDev^2));
            rangeGaussian = exp(-(localPatch - inputImage(y, x)).^2 / (2 * rangeStdDev^2));

            filteredImage(y, x) = sum(sum(spatialGaussian .* rangeGaussian .* localPatch)) / sum(sum(spatialGaussian .* rangeGaussian));
        end
    end
end
