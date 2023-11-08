function denoised_image = myPCADenoising2(im1, patch_size, neighbourhood_size, sigma)
    [M, N] = size(im1);
    PatchImage = []; % Matrix to store denoised patches

    % Iterate over the image to get patches
    for J = 1:N - patch_size + 1
        for I = 1:M - patch_size + 1
            % Extract the current patch
            P = im1(I:I + patch_size - 1, J:J + patch_size - 1);

            % Define the neighborhood boundaries
            Top = max(1, I - floor(neighbourhood_size / 2));
            Bottom = min(M - patch_size + 1, I + floor(neighbourhood_size / 2));
            Left = max(1, J - floor(neighbourhood_size / 2));
            Right = min(N - patch_size + 1, J + floor(neighbourhood_size / 2));

            % Extract patches from the neighborhood
            PatchNeighbourhood = im1(Top:Bottom, Left:Right);
            Patches = im2col(PatchNeighbourhood, [patch_size, patch_size], 'sliding');

            % Calculate distances between patches
            Distances = sum((Patches - P(:)).^2);

            % Sort distances
            [~, SortedIndices] = sort(Distances);

            % Limit the number of patches to consider
            K = min(200, size(Distances, 2));
            SelectedPatches = Patches(:, SortedIndices(1:K));
            N = size(SelectedPatches, 2);

            % Compute eigen vectors of the selected patches
            [EigenVectors, ~] = eig(SelectedPatches * SelectedPatches');

            % Calculate coefficients for each patch
            Coefficients = EigenVectors' * SelectedPatches;
            CentralPatchCoefficients = EigenVectors' * P(:);

            % Calculate alpha values for each patch
            AlphaI = max(0, (1 / N) * (sum(Coefficients.^2, 2)) - sigma^2);

            % Update the alpha value for the central patch
            for Index = 1:patch_size^2
                CentralPatchCoefficients(Index) = CentralPatchCoefficients(Index) / (1 + sigma^2 / AlphaI(Index));
            end

            % Reconstruct the denoised patch
            DenoisedPatch = EigenVectors * CentralPatchCoefficients;

            % Concatenate the denoised patches
            PatchImage = [PatchImage, DenoisedPatch];
        end
    end

    [M, N] = size(im1);

    % Create indices for patches
    Indices = reshape(1:M * N, [M, N]);
    Subscripts = im2col(Indices, [7, 7], 'sliding');

    % Average overlapping patches to reconstruct the denoised image
    denoised_image = accumarray(Subscripts(:), PatchImage(:)) ./ accumarray(Subscripts(:), 1);

    % Rescale the denoised image to the range of the original image
    denoised_image = reshape(denoised_image, M, N);
    denoised_image = (denoised_image - min(denoised_image(:))) / (max(denoised_image(:)) - min(denoised_image(:))) * 255;
end
