function filtered_I = mean_shift_filter(I, sig_s, sig_r)
    [r, c] = size(I);
    BWby2 = ceil(3 * sig_s) + 1;
    epsilon = 0.01;
    filtered_I = zeros(size(I));
    
    for x = 1:c
        for y = 1:r
            feature = [x, y, I(y, x)]; 
            
            while true
                i1 = max(y - BWby2, 1); i2 = min(y + BWby2, r);
                j1 = max(x - BWby2, 1); j2 = min(x + BWby2, c);
                
                local_intensity = I(i1:i2, j1:j2);
                [X, Y] = meshgrid(j1:j2, i1:i2);
                
                Gs = exp(-((X - feature(1)).^2 + (Y - feature(2)).^2) / (2 * sig_s^2));
                Gr = exp(-((local_intensity - feature(3)).^2) / (2 * sig_r^2));
                G = Gs .* Gr;
                
                Wp = sum(G, 'all');
                fx = sum(G .* X, 'all') / Wp;
                fy = sum(G .* Y, 'all') / Wp;
                fI = sum(G .* local_intensity, 'all') / Wp;
                
                if norm(feature - [fx, fy, fI]) > epsilon
                    feature = [fx, fy, fI];
                else
                    break;
                end
            end
            
            filtered_I(y, x) = feature(3);
        end
    end
end
