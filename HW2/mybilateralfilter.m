function filtered_image = mybilateralfilter(input_image, spatial_sigma, range_sigma)
    % Get the size of the input image
    [M, N] = size(input_image);
    
    % Calculate the filter window size
    window_size = ceil(3 * spatial_sigma);
    
    % Initialize the output filtered image
    filtered_image = zeros(M, N);

    for row = 1:M
        for col = 1:N
            % Define the boundaries of the local window
            row_start = max(row - window_size, 1);
            row_end = min(row + window_size, M);
            col_start = max(col - window_size, 1);
            col_end = min(col + window_size, N);
            
            % Extract the region of interest (ROI)
            local_region = input_image(row_start:row_end, col_start:col_end);
            
            % Calculate the intensity difference
            intensity_diff = local_region - input_image(row, col);

            % Calculate the spatial Gaussian kernel
            spatial_kernel = gaussian_kernel(2 * window_size + 1, 2 * window_size + 1, spatial_sigma);
            
            % Calculate the range Gaussian kernel
            range_kernel = gaussian_kernel(row_end - row_start + 1, col_end - col_start + 1, range_sigma);

            % Compute the weighted sum
            weighted_sum = sum(sum(spatial_kernel .* range_kernel .* local_region));
            
            % Compute the normalization factor
            normalization_factor = sum(spatial_kernel(:) .* range_kernel(:));

            % Update the filtered pixel value
            filtered_image(row, col) = round(weighted_sum / normalization_factor);
        end
    end
end

function gaussian_kernel = gaussian_kernel(m, n, sigma)
    % Create a 2D Gaussian kernel
    [X, Y] = meshgrid(-(n - 1) / 2:(n - 1) / 2, -(m - 1) / 2:(m - 1) / 2);
    exponent = -(X.^2 + Y.^2) / (2 * sigma^2);
    gaussian_kernel = exp(exponent);
    
    % Normalize the kernel
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));
end