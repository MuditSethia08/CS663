im = double(imread('barbara256.png'));

sample = [1 2 3; 4 5 6; 7 8 9];

X = im2col(sample, [2,2], "sliding");

[V,~] = eig(X*X');