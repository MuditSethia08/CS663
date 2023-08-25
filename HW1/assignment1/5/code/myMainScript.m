%% MyMainScript

tic;
% Read the images
im1 = double(imread("goi1.jpg"));
im2 = double(imread("goi2_downsampled.jpg"));

% Display the images
figure;
imshow(im1);
title('Image 1');
figure;
imshow(im2);
title('Image 2');

% Initialize arrays to store corresponding points
x1 = zeros(12, 1);
y1 = zeros(12, 1);
x2 = zeros(12, 1);
y2 = zeros(12, 1);

% Collect corresponding points
for i = 1 : 3
    figure(1); 
    imshow(im1/255);
    [x1(i), y1(i)] = ginput(1);
    figure(2); 
    imshow(im2/255); 
    [x2(i), y2(i)] = ginput(1);
end

points_img1 = [x1, y1];
points_img2 = [x2, y2];

% Compute the affine transformation matrix
transformation = fitgeotrans(points_img1, points_img2, 'affine');

% Apply the transformation to image 2 to align it with image 1
outputImage = imwarp(im2, transformation);

% % Display the aligned images
% figure;
% imshowpair(im1, outputImage, 'montage');
% title('Image 1 (Left) and Aligned Image 2 (Right)');

% Display the affine transformation matrix
disp('Affine Transformation Matrix:');
disp(transformation.T);

toc;
