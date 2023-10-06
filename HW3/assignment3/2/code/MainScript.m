
tic;
clear; close all; clc;
img = im2double(imread('barbara256.png'));
figure; imshow(img); colormap("gray"); title('Original Image');

padded_img = padarray(img, [ceil(size(img, 1) / 2), ceil(size(img, 2) / 2)]); %padding the image
%padded_img = img;

 
img_filtered_ideal_40 = IdealLowPass_filter(padded_img, 40); % Ideal Low pass filtered image with cutoff frequency D = 40
img_filtered_ideal_80 = IdealLowPass_filter(padded_img, 80);  % Ideal Low pass filtered image with cutoff frequency D = 80
img_filtered_gaussian_40 = GaussianLowPass_filter(padded_img, 40);  % Gaussian Low Pass filtered image with sigma = 40
img_filtered_gaussian_80 = GaussianLowPass_filter(padded_img, 80);  % Gaussian Low Pass filtered image with sigma = 80

figure;   %display
subplot(1, 2, 1);
imshow(img_filtered_ideal_40); title('Ideal LPF D = 40');
subplot(1, 2, 2);
imshow(img_filtered_ideal_80); title('Ideal LPF D = 80');

figure;  %display
subplot(1, 2, 1);
imshow(img_filtered_gaussian_40); title('Gauss LPF \sigma = 40');
subplot(1, 2, 2);
imshow(img_filtered_gaussian_80); title('Gauss LPF \sigma = 80');

% Display the Fourier transform images of the original and filtered images
figure;
subplot(2, 3, 1); imshow(log(abs(fftshift(fft2(img))) + 1), []); title('FT of Original Image');
subplot(2, 3, 2); imshow(log(abs(fftshift(fft2(img_filtered_ideal_40))) + 1), []); title('FT of Ideal LPF (D = 40)');
subplot(2, 3, 3); imshow(log(abs(fftshift(fft2(img_filtered_gaussian_40))) + 1), []); title('FT of Gauss LPF (\sigma = 40)');
subplot(2, 3, 5); imshow(log(abs(fftshift(fft2(img_filtered_ideal_80))) + 1), []); title('FT of Ideal LPF (D = 80)');
subplot(2, 3, 6); imshow(log(abs(fftshift(fft2(img_filtered_gaussian_80))) + 1), []); title('FT of Gauss LPF (\sigma = 80)');

toc;


function img_filtered = GaussianLowPass_filter(img, sigma)
    Fourier_Transformed = fftshift(fft2(img));%find fourier transform and centre shift

    % Create the Gaussian low pass filter
    [x, y] = meshgrid(-size(Fourier_Transformed, 1) / 2:size(Fourier_Transformed, 1) / 2 - 1, -size(Fourier_Transformed, 2) / 2:size(Fourier_Transformed, 2) / 2 - 1);
    Filter = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    img_filtered = ifft2(ifftshift(Fourier_Transformed .* Filter));

end

function img_filtered = IdealLowPass_filter(img, cutoff_freq)
    
    Fourier_Transformed = fftshift(fft2(img));  %find fourier transform and centre shift

    % Apply the low pass filter of cutoff frequency
    Filter = zeros(size(Fourier_Transformed));
    [x, y] = meshgrid(-size(Filter, 1) / 2:size(Filter, 1) / 2 - 1, -size(Filter, 2) / 2:size(Filter, 2) / 2 - 1);

    %apply the low pass filter condition
    indices = (x.^2 + y.^2) <= cutoff_freq^2;
    Filter(indices) = 1;
    img_filtered = ifft2(ifftshift(Fourier_Transformed .* Filter));

end
