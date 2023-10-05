%% MyMainScript

tic;
%% Your code here

clear; close all;

%% Input images

barbaraImage = im2double(imread('barbara256.png'));
kodakImage = im2double(imread('kodak24.png'));

figure(1); imagesc(barbaraImage); colormap("gray"); title("Original barbara256");
figure(2); imagesc(kodakImage); colormap("gray"); title("Original kodak24");

%% Noisy images

noiseStdDev = 10 / 255; % noise standard deviation

noisyBarbara = barbaraImage + noiseStdDev * randn(size(barbaraImage));
noisyKodak = kodakImage + noiseStdDev * randn(size(kodakImage));

figure(3); imagesc(noisyBarbara); colormap("gray"); title("Noisy barbara256 with \sigma_n = " + num2str(noiseStdDev * 255));
figure(4); imagesc(noisyKodak); colormap("gray"); title("Noisy kodak24 with \sigma_n = " + num2str(noiseStdDev * 255));

%% Mean shift filtering

sig_s = 2;
sig_r = 2;
barbaraFiltered = mean_shift_filter(noisyBarbara, sig_s, sig_r / 255);
kodakFiltered = mean_shift_filter(noisyKodak, sig_s, sig_r / 255);

figure(5); imagesc(barbaraFiltered); colormap("gray"); 
title("Mean shifted filter on barbara256 with \sigma_n = " + num2str(noiseStdDev * 255) + ", \sigma_s = " + num2str(sig_s) + ", \sigma_r = " + num2str(sig_r));
figure(6); imagesc(kodakFiltered); colormap("gray"); 
title("Mean shifted filter on kodak24 with \sigma_n = " + num2str(noiseStdDev * 255) + ", \sigma_s = " + num2str(sig_s) + ", \sigma_r = " + num2str(sig_r));

toc;
