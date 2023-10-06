z = zeros(201,201) ; 
z(101 , : ) = 255 ;

figure(1) ; imshow(z);
FA = fftshift(fft2(z));
figure(2) ; imshow(FA);
figure(3) ; imshow(log(FA+1));
