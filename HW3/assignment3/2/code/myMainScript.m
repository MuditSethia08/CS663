%% MyMainScript

tic;
%% Your code here
% Clear workspace and close all figures
clear; close all; clc;

% Read the barbara256.png image as a double
img = im2double(imread('barbara256.png'));

% Display the original image
figure; imshow(img); colormap("gray"); title('Original Image');

% Pad the image to make the dimensions twice as large
img_padded = padarray(img, [size(img, 1) / 2, size(img, 2) / 2]);

% Ideal Low pass filtered image with cutoff frequency D = 40
cutoff_freq_ideal = 40;
img_filtered_ideal_40 = IdealLowPass(img_padded, cutoff_freq_ideal);

% Ideal Low pass filtered image with cutoff frequency D = 80
cutoff_freq_ideal = 80;
img_filtered_ideal_80 = IdealLowPass(img_padded, cutoff_freq_ideal);

% Display filtered images when using an ideal Low Pass Filter
figure;
subplot(1, 2, 1);
imshow(img_filtered_ideal_40); colormap("gray"); title('Ideal LPF (D = 40)');
subplot(1, 2, 2);
imshow(img_filtered_ideal_80); colormap("gray"); title('Ideal LPF (D = 80)');

% Gaussian Low Pass filtered image with sigma = 40
sigma_gaussian = 40;
img_filtered_gaussian_40 = GaussianLowPass(img_padded, sigma_gaussian);

% Gaussian Low Pass filtered image with sigma = 80
sigma_gaussian = 80;
img_filtered_gaussian_80 = GaussianLowPass(img_padded, sigma_gaussian);

% Display filtered images when using a Gaussian Low Pass Filter
figure;
subplot(1, 2, 1);
imshow(img_filtered_gaussian_40); colormap("gray"); title('Gaussian LPF (\sigma = 40)');
subplot(1, 2, 2);
imshow(img_filtered_gaussian_80); colormap("gray"); title('Gaussian LPF (\sigma = 80)');

% Display the Fourier transform images of the original and filtered images
figure;
subplot(2, 3, 1);
imshow(log(abs(fftshift(fft2(img))) + 1), []); colormap("jet"); title('FT of Original Image');

subplot(2, 3, 2);
imshow(log(abs(fftshift(fft2(img_filtered_ideal_40))) + 1), []); colormap("jet"); title('FT of Ideal LPF (D = 40)');

subplot(2, 3, 3);
imshow(log(abs(fftshift(fft2(img_filtered_ideal_80))) + 1), []); colormap("jet"); title('FT of Ideal LPF (D = 80)');

subplot(2, 3, 5);
imshow(log(abs(fftshift(fft2(img_filtered_gaussian_40))) + 1), []); colormap("jet"); title('FT of Gaussian LPF (\sigma = 40)');

subplot(2, 3, 6);
imshow(log(abs(fftshift(fft2(img_filtered_gaussian_80))) + 1), []); colormap("jet"); title('FT of Gaussian LPF (\sigma = 80)');

%% Define IdealLowPass and GaussianLowPass functions here

toc;
function img_filtered = IdealLowPass(img, cutoff_freq)
    % Compute the Fourier transform of the image along with shift
    F = fftshift(fft2(img));

    % Apply the low pass filter of cutoff frequency
    Filter = zeros(size(F));
    [x, y] = meshgrid(-size(Filter, 1) / 2:size(Filter, 1) / 2 - 1, -size(Filter, 2) / 2:size(Filter, 2) / 2 - 1);
    valid_indices = (x.^2 + y.^2) <= cutoff_freq^2;
    Filter(valid_indices) = 1;

    % Filtering the image
    F_filtered = F .* Filter;

    img_filtered = ifft2(ifftshift(F_filtered));
end

function img_filtered = GaussianLowPass(img, sigma)
    % Compute the Fourier transform of the image along with shift
    F = fftshift(fft2(img));

    % Create the Gaussian low pass filter
    [x, y] = meshgrid(-size(F, 1) / 2:size(F, 1) / 2 - 1, -size(F, 2) / 2:size(F, 2) / 2 - 1);
    Filter = exp(-(x.^2 + y.^2) / (2 * sigma^2));

    % Filtering
    F_filtered = F .* Filter;

    img_filtered = ifft2(ifftshift(F_filtered));
end
