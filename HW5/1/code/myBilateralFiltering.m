
 function filtered_img = myBilateralFiltering(input_image, filter_size, space_sigma, intensity_sigma)
    output_image = double(zeros(size(input_image, 1), size(input_image, 2)));

    [X, Y] = meshgrid(-floor(filter_size / 2):floor(filter_size / 2), -floor(filter_size / 2):floor(filter_size / 2));
    spatial_weights = exp(-(X.^2 + Y.^2) / (2 * space_sigma^2));

    wait = waitbar(0, "Bilateral Filter in Progress");

    for i = 1:size(input_image, 1)
        i_min = max(i - floor(filter_size / 2), 1);
        i_max = min(i + floor(filter_size / 2), size(input_image, 1);

        for j = 1:size(input_image, 2)
            j_min = max(j - floor(filter_size / 2), 1);
            j_max = min(j + floor(filter_size / 2), size(input_image, 2);

            curr_intensity = input_image(i, j);
            intensity_window = input_image(i_min:i_max, j_min:j_max);

            intensity_weights = exp(-(intensity_window - curr_intensity).^2 / (2 * intensity_sigma^2));

            overall_weights = spatial_weights((i_min:i_max) - i + floor(filter_size / 2) + 1, (j_min:j_max) - j + floor(filter_size / 2) + 1) .* intensity_weights;

            output_image(i, j) = sum(sum(overall_weights .* intensity_window)) / sum(sum(overall_weights));
        end

        waitbar(i / double(size(input_image, 1)));
    end

    close(wait);
end
