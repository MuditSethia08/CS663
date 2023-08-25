%% MyMainScript

tic;
% Read the images
im1 = double(imread("goi1.jpg"));
im2 = double(imread("goi2_downsampled.jpg"));

%%
% Display the images
figure;
imshow(im1/255);
title('Image 1');
figure;
imshow(im2/255);
title('Image 2');

%%
% Collect corresponding points
% for i = 1 : 3
%     figure(1); 
%     imshow(im1/255);
%     [x1(i), y1(i)] = ginput(1);
%     figure(2); 
%     imshow(im2/255); 
%     [x2(i), y2(i)] = ginput(1);
% end
% 
% 

x1 = [266   204   352];
y1 = [78.0000  256.0000  256.0000];
x2 = [304   238   392];
y2 = [96.0000  276.0000  274.0000];

points1 = [x1', y1'];
points2 = [x2', y2'];

%%
len = size(x1);
init = [x1;y1;ones(len)];
fin = [x2;y2;ones(len)];

% Calculating the Transformation using the least-sqaure framework.

tform = (fin*(init'))/(init*(init'));


% Display the affine transformation matrix
disp('Affine Transformation Matrix:');
disp(tform);

%%
% Compute the affine transformation matrix
tform_func = fitgeotrans(points2, points1, 'affine');

% Apply the transformation to image 2 to align it with image 1
outputImage = imwarp(im2, tform_func);

%% 
% For nearest neighbour interpolation
dim_og = size(im1);
new_im = zeros(dim_og);
inv_tform = inv(tform);
for x = 1:dim_og(1)
    for y = 1:dim_og(2)
        corr_im = round(tform\[x,y,1]');
        n_x = corr_im(1);
        n_y = corr_im(2);
        if(n_x > dim_og(1) || n_x <1)
            continue
        end
        if(n_y > dim_og(2) || n_y <1)
            continue
        end
        new_im(x, y) = im1(n_x,n_y)/255;
    end
end

%%
% Bilinear interpolation
for x = 1:dim_og(1)
    for y = 1:dim_og(2)
        % Apply the inverse transformation to find the corresponding point
        corr_im = inv_tform * [x; y; 1];
        n_x = corr_im(1);
        n_y = corr_im(2);
        
        % Check if the transformed point is within bounds
        if (n_x > dim_og(1) || n_x < 1 || n_y > dim_og(2) || n_y < 1)
            continue;
        end
        
        % Calculate the four surrounding pixel coordinates
        x1 = floor(n_x);
        x2 = ceil(n_x);
        y1 = floor(n_y);
        y2 = ceil(n_y);
        
        % Compute the fractional parts for interpolation
        alpha = n_x - x1;
        beta = n_y - y1;
        
        % Perform bilinear interpolation
        new_im_bilinear(x, y) = (1 - alpha) * ((1 - beta) * im1(x1, y1) + beta * im1(x1, y2)) + ...
            alpha * ((1 - beta) * im1(x2, y1) + beta * im1(x2, y2));
    end
end

%% 
% Display the aligned images
figure;
montage({im1/255, im2/255, new_im, new_im_bilinear/255});
toc;

%%
clc
