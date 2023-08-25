clear; clc;

tic;
im1 = im2double(imread('goi1.jpg')); figure(1); imshow(im1); title('goi1');
im2 = im2double(imread('goi2_downsampled.jpg')); figure(2); imshow(im2); title('goi2');

% control points
%for i=1:12
%    figure(1); imshow(im1); [x1(i), y1(i)] = ginput(1);
%    figure(2); imshow(im2); [x2(i), y2(i)] = ginput(1);
%end
x1 = [571,433,419,373,264,105,68,278,459,514,265,95];
y1 = [175,164,108,23,79,30,232,256,233,300,160,88];
x2 = [635,476,463,415,303,145,104,314,507,574,298,135];
y2 = [184,176,120,35,97,57,255,275,249,319,179,112];
x1 = round(x1); x2 = round(x2); y1 = round(y1); y2 = round(y2);

% P2 = AP1, P2 is of second image, A is affine matrix, P1 is of first image
% A = (P2P1')(P1P1')^-1
P1 = [x1;y1;ones(1,length(x1))];
P2 = [x2;y2;ones(1,length(x2))];
A = (P2*(P1'))/(P1*(P1'));

% reverse warping with nearest neighbour interpolation
TNN = zeros(size(im1));
[X,Y] = meshgrid(1:size(im1,2),1:size(im1,1));
V = [X(:)';Y(:)';ones(1,numel(im1))];
U = A\V;
U = round(U); % nearest neighbour
X = U(1,:); Y = U(2,:); b = and(and(X>0,X<=640),and(Y>0,Y<=360));
U1D = Y(b) + (size(im1,1)*(X(b)-1));
X = V(1,:); Y = V(2,:);
V1D = Y(b) + (size(im1,1)*(X(b)-1));
TNN(V1D) = im1(U1D); figure(3); imshow(TNN); title('Transformed image goi1 with nearest neighbour');

% reverse warping with bilinear interpolation
TBL = zeros(size(im1));
[X,Y] = meshgrid(1:size(im1,2),1:size(im1,1));
V = [X(:)';Y(:)';ones(1,numel(im1))];
U = A\V;
X = U(1,:); Y = U(2,:); b = and(and(X>=1,X<=640),and(Y>=1,Y<=360));
bilip = bilinear_ip(im1,X(b),Y(b));
X = V(1,:); Y = V(2,:);
V1D = Y(b) + (size(im1,1)*(X(b)-1));
TBL(V1D) = bilip; figure(4); imshow(TBL); title('Transformed image goi1 with bilinear interpolation');

figure;
montage({im1, im2, TBL});
toc;

function bilip = bilinear_ip(I,X,Y)
    LB = floor(Y) + (size(I,1)*(floor(X)-1));
    LBw = (ceil(X)-X).*(ceil(Y)-Y);
    RB = floor(Y) + (size(I,1)*(ceil(X)-1));
    RBw = (X-floor(X)).*(ceil(Y)-Y);
    LT = ceil(Y) + (size(I,1)*(floor(X)-1));
    LTw = (ceil(X)-X).*(Y-floor(Y));
    RT = ceil(Y) + (size(I,1)*(ceil(X)-1));
    RTw = (X-floor(X)).*(Y-floor(Y));
    bilip = I(LB).*LBw + I(RB).*RBw + I(LT).*LTw + I(RT).*RTw;
end