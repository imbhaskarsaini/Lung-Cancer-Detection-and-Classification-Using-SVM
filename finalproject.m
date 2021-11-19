clc;
clear all;
close all;
% read the input CT image
I=imread('C:\Users\imbha\OneDrive\Desktop\image5c.png');
figure
imshow(I);
title('INPUT CT IMAGE');
Igra1=rgb2gray(I);
figure
imshow(Igra1);
title('GRAY IMAGE');
Ifil2=medfilt2(Igra1,[3,3]);
 figure
imshow(Ifil2);
 title('FILTERED GRAY IMAGE');
text(732,501,'Image courtesy of Corel(R)',...
     'FontSize',7,'HorizontalAlignment','right')
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(Ifil2), hy, 'replicate');
Ix = imfilter(double(Ifil2), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
figure
imshow(gradmag,[]), title('Gradient magnitude (gradmag)')
L = watershed(gradmag);
Lrgb = label2rgb(L);
figure, imshow(Lrgb), title('Watershed transform of gradient magnitude (Lrgb)')
se = strel('disk', 20);
Io = imopen(Ifil2, se);
figure
imshow(Io), title('Opening (Io)')
Ie = imerode(Ifil2, se);
Iobr = imreconstruct(Ie, Ifil2);
figure
imshow(Iobr), title('Opening-by-reconstruction (Iobr)')
Ioc = imclose(Io, se);
figure
imshow(Ioc), title('Opening-closing (Ioc)')
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
figure
imshow(Iobrcbr), title('Opening-closing by reconstruction (Iobrcbr)')
fgm = imregionalmax(Iobrcbr);
figure
imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')
I21 = Ifil2;
Ifil2(fgm) = 255;
figure
imshow(I21), title('Regional maxima superimposed on original image (I21)')
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);
fgm4 = bwareaopen(fgm3, 20);
I3 = Ifil2;
I3(fgm4) = 255;
figure
imshow(I3)
title('Modified regional maxima superimposed on original image (fgm4)')
bw = imbinarize(Iobrcbr);
figure
imshow(bw), title('Thresholded opening-closing by reconstruction (bw)')
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
figure
imshow(bgm), title('Watershed ridge lines (bgm)')
gradmag2 = imimposemin(gradmag, bgm | fgm4);
L = watershed(gradmag2);
I4 = Ifil2;
I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
figure
imshow(I4)
title('Markers and object boundaries superimposed on original image (I4)')
Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
figure
imshow(Lrgb)
title('Colored watershed label matrix (Lrgb)')
figure
imshow(Ifil2)
hold on
himage = imshow(Lrgb);
himage.AlphaData = 0.3;
title('Lrgb superimposed transparently on original image')
x = double(fgm);
m = size(fgm,1);
n = size(fgm,2);
signal1 = fgm(:,:);
%Feat = getmswpfeat(signal,winsize,wininc,J,'matlab');
%Features = getmswpfeat(signal,winsize,wininc,J,'matlab');
[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');
DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
 feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
%%SVM TRAINING AND CLASSIFICATION
 database = xlsread('datacan.xls');   
% Read Database in excel file
disp('data base of patients ');
disp(database);
Contrast = database(1:9,1);
Correlation = database(1:9,2);
Energy = database(1:9,3);
Homogeneity = database(1:9,4);
Mean = database(1:9,5);
Standard_Deviation = database(1:9,6);
Entropy = database(1:9,7);
RMS = database(1:9,8);
Variance = database(1:9,9);
Smoothness = database(1:9,10);
Kurtosis = database(1:9,11);
Skewness = database(1:9,12);
IDM = database(1:9,13);
xdata = ([Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM]);
group = database(1:9,14);
svmStruct= fitcsvm(xdata,group,'ClassNames',[1,0],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',inf);
% Classifying New input data
disease = predict(svmStruct,[feat]);

if (disease>=0.16)
disp('Patient is having cancer');
h=msgbox('Patient is having cancer','RESULT','custom',I);
else
disp('Patient is not having cancer');
 h=msgbox('Patient is not having cancer','RESULT','custom',I);
end
