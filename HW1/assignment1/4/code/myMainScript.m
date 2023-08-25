%% MyMainScript

tic; % Start measuring execution time

%% Your Code Here

% Load images T1.jpg and T2.jpg and convert them to double precision
J1 = im2double(imread('T1.jpg'));
J2 = im2double(imread('T2.jpg'));

% Rotate image J2 by 28.5 degrees anticlockwise and crop
J3 = imrotate(J2, 28.5, 'crop');

% Define a range of angles to test for alignment
angles = -45:1:45;

% Initialize arrays to store scores for different alignment metrics
NCC = zeros(1, length(angles)); % Normalized Cross Correlation
JE = zeros(1, length(angles));  % Joint Entropy
QMI = zeros(1, length(angles)); % Quadrature Mutual Information

% Set the width of histogram bins
bin_width = 10;

%% Loop through Different Rotation Angles

for i = 1:length(angles)
    % Rotate the cropped J2 image by the current angle
    J4 = imrotate(J3 + 1, angles(i), 'crop');  % Adding 1 to remember valid pixels
    J4 = J4 - 1; % Subtracting 1 to revert to original values
    
    % Find pixels that are valid in both images after rotation
    valid_pixels = J4 ~= -1;
    
    % Extract valid pixel values from the original and rotated images
    image1_valid = J1(valid_pixels);
    rotated_image_valid = J4(valid_pixels);

    % Compute joint histogram and alignment scores
    joint_histogram = compute_joint_histogram(image1_valid, rotated_image_valid, bin_width);
    NCC(i) = compute_normalized_cross_correlation(image1_valid, rotated_image_valid);
    JE(i) = compute_joint_entropy(joint_histogram);
    QMI(i) = compute_quadrature_mutual_information(joint_histogram);
end

%% Plot Results

% Plot NCC score versus angle
figure(1); plot(angles, NCC); xlabel('Angle'); ylabel('NCC Score'); title('Normalized Cross Correlation'); grid on;

% Plot JE score versus angle
figure(2); plot(angles, JE); xlabel('Angle'); ylabel('JE Score'); title('Joint Entropy'); grid on;

% Plot QMI score versus angle
figure(3); plot(angles, QMI); xlabel('Angle'); ylabel('QMI Score'); title('Quadrature Mutual Information'); grid on;

% Find the angle that minimizes joint entropy
[minimum_je, index] = min(JE);

% Rotate the cropped J2 image to the optimized angle
rotated_image_optimized = imrotate(J3 + 1, angles(index), 'crop');
rotated_image_optimized = rotated_image_optimized - 1;

% Find valid pixels in the optimized rotated image
valid_pixels_optimized = rotated_image_optimized ~= -1;

% Extract valid pixel values from the original and optimized rotated images
image1_valid_optimized = J1(valid_pixels_optimized);
rotated_image_valid_optimized = rotated_image_optimized(valid_pixels_optimized);

% Compute joint histogram for the optimized alignment
joint_histogram_optimized = compute_joint_histogram(image1_valid_optimized, rotated_image_valid_optimized, bin_width);

% Plot the joint histogram for the optimized alignment
figure(4); imagesc(0:bin_width:255, 0:bin_width:255, joint_histogram_optimized); colorbar; xlabel('J2'); ylabel('J1'); title('Joint Histogram for Minimum JE with Bin Width 10');

% Display the original and intermediate images
figure(5); imshow(J1); title('Image 1');
figure(6); imshow(J2); title('Image 2');
figure(7); imshow(J3); title('Rotated Image 2 by 28.5 Degrees Anti-Clockwise');
figure(8); imshow(rotated_image_optimized); title('Aligned Rotated Image 2 with Image 1');

toc; % Stop measuring execution time

% Define helper functions for alignment metrics and histogram computation

function ncc = compute_normalized_cross_correlation(image1, image2)
    % Compute normalized cross-correlation between two images
    mean_image1 = mean(image1, 'all');
    mean_image2 = mean(image2, 'all');
    numerator = sum((image1 - mean_image1) .* (image2 - mean_image2), 'all');
    denominator = sqrt(sum((image1 - mean_image1).^2, 'all') * sum((image2 - mean_image2).^2, 'all'));
    ncc = abs(numerator / denominator);
end

function je = compute_joint_entropy(joint_histogram)
    % Compute joint entropy from a joint histogram
    valid_bins = joint_histogram ~= 0;
    je = -sum(joint_histogram(valid_bins) .* log2(joint_histogram(valid_bins)), 'all');
end

function qmi = compute_quadrature_mutual_information(joint_histogram)
    % Compute quadrature mutual information from a joint histogram
    marginal_histogram1 = sum(joint_histogram, 2);
    marginal_histogram2 = sum(joint_histogram, 1);
    qmi = sum((joint_histogram - marginal_histogram1 * marginal_histogram2).^2, 'all');
end

function joint_histogram = compute_joint_histogram(image1, image2, bin_width)
    % Compute joint histogram between two images with a specified bin width
    bins = ceil(255 / bin_width);
    bin_image1 = floor(image1 .* (255 / bin_width)) + 1;
    bin_image2 = floor(image2 .* (255 / bin_width)) + 1;
    joint_histogram = accumarray([bin_image1(:), bin_image2(:)], ones(1, length(image1(:))));
    joint_histogram = joint_histogram ./ (length(image1(:)));
end
