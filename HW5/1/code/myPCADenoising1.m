function denoised_img=myPCADenoising1(im1,sigma)
    
    patch = [7 7];

    %% converting patches into column vectors to get the matrix P
    X = im2col(im1,patch,'sliding') ;   
    [V,~] = eig(X*X');                  
    eigen_coeff_vec = V'*X;             

    [~,N] = size(eigen_coeff_vec); 
    right = (sum(eigen_coeff_vec.*eigen_coeff_vec,2))/N - sigma*sigma;
    average = max(0,right);
    den = eigen_coeff_vec./(1+(sigma*sigma)./average); 
    qdenoise = V*den; 
    newsize = size(im1) - [7,7] +1;
    denoised_img_distinct = col2im(qdenoise,[7,7],newsize.*[7,7],'distinct');
    denoised_img_temp = zeros([size(denoised_img_distinct,1),size(im1,2)]);

    denoised_img = zeros(size(im1));
    denom_row_temp = ones(1,size(denoised_img_distinct,2));
    denom_col_temp = ones(size(denoised_img_distinct,1),1);
    denom_row = zeros(1,size(im1,2));
    denom_col = zeros(size(im1,1),1);

    for i=1:newsize(2)
            denoised_img_temp(:,i:i+patch(2)-1) = denoised_img_temp(:,i:i+patch(2)-1) + denoised_img_distinct(:,(i-1)*patch(2)+1:i*patch(2));
            denom_row(1,i:i+patch(2)-1) = denom_row(1,i:i+patch(2)-1) + denom_row_temp(1,(i-1)*patch(2)+1:i*patch(2));
    end
    for i=1:newsize(1)
            denoised_img(i:i+patch(1)-1,:) = denoised_img(i:i+patch(1)-1,:) + denoised_img_temp((i-1)*patch(1)+1:i*patch(1),:);
            denom_col(i:i+patch(1)-1,1) = denom_col(i:i+patch(1)-1,1) + denom_col_temp((i-1)*patch(1)+1:i*patch(1),:);
    end

    denom = denom_col*denom_row;
    denoised_img = denoised_img./denom;
end