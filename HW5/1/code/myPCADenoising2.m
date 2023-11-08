function denoised_image = myPCADenoising2(im1, patch_size, neighbourhood_size, sigma)
    % Iterate over the image to get the patches
    [m, n] = size(im1);
    P_image = []; % Matrix to store the patches of the image after denoising

    for j = 1:n - patch_size + 1

        for i = 1:m - patch_size + 1
            patch = im1(i:i + patch_size - 1, j:j + patch_size - 1);
            % 31 x 31 neighbourhood centered at (i, j) in the image
            neighbourhood_top = max(1, i - floor(neighbourhood_size / 2));
            neighbourhood_bottom = min(m - patch_size + 1, i + floor(neighbourhood_size / 2));
            neighbourhood_left = max(1, j - floor(neighbourhood_size / 2));
            neighbourhood_right = min(n - patch_size + 1, j + floor(neighbourhood_size / 2));
            patch_neighbourhood = im1(neighbourhood_top:neighbourhood_bottom, neighbourhood_left:neighbourhood_right);

            % Make 7x7 patches in the patch_neighbourhood
            patch_neighbourhood_patches = im2col(patch_neighbourhood, [patch_size, patch_size], 'sliding');

            % Find the distance between the patch and each of the patches in the neighbourhood
            distances = sum((patch_neighbourhood_patches - patch(:)).^2);

            % Sort the distances
            [~, sorted_indices] = sort(distances);

            K = min(200, size(distances, 2));
            % Take patch vectors corresponding to 200 minimum distances
            P = patch_neighbourhood_patches(:, sorted_indices(1:K));

            N = size(P, 2);

            % Compute eigen vector matrix of the PP' matrix
            [V, D] = eig(P * P');

            % Eigen coefficients for each of the patches
            alphas = V' * P;

            % Eigen coefficients for the central patch
            alpha_central_patch = V' * patch(:);

            % Now we update the alpha value for the central patch
            alpha_i = max(0, (1 / N) * (sum(alphas.^2, 2)) - sigma^2);

            % Update the alpha value for the central patch
            for i = 1:patch_size^2
                alpha_central_patch(i) = alpha_central_patch(i) / (1 + sigma^2 / alpha_i(i));
            end

            % Reconstruct the reference patches
            P_denoised = V * alpha_central_patch;

            P_image = [P_image, P_denoised]; % Concatenate the denoised patches
        end

    end

    % Reconstruct the image from the denoised patches
    % Reference code used from https://in.mathworks.com/matlabcentral/answers/157285-averaging-overlapping-pixels-in-sliding-window-operation

    [m, n] = size(im1);

    indices = reshape(1:m * n, [m, n]);
    subs = im2col(indices, [7, 7], 'sliding');

    denoised_image = accumarray(subs(:), P_image(:)) ./ accumarray(subs(:), 1);
    denoised_image = reshape(denoised_image, m, n);

    % Scale the denoised image to the range of the original image
    denoised_image = (denoised_image - min(denoised_image(:))) / (max(denoised_image(:)) - min(denoised_image(:))) * 255;
end