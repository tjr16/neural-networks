function [X, Y, y] = LoadBatch(filename)
% A function that reads in the data from 
% a CIFAR-10 batch file and returns the 
% image and label data in separate files. 
% ----------
% Return:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   Y: k X n (10 X 10000)
%      uint8, {0, 1}
%   y: 1 X n (1 X 10000)
%      uint8, {1, 2, ..., 10}

    A = load(filename);
    % data
    I = reshape(A.data', 32, 32, 3, 10000);
    I = permute(I, [2, 1, 3, 4]);
    I = reshape(I, [], 10000);
    X = double(I) ./255;
    % label
    L = A.labels';
    Y = onehotencode(categorical(L), 1);
    y = L + 1;

end