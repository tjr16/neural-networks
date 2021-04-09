function [train_data, valid_data, test_data] = loadData()
% A function that loads and preprocesses training, validation
% and test datasets.
% ----------
% Return:
%   train_data: cell, 1 X 3
%       X: image pixel data, d X n (3072 X 10000) 
%           double, [0, 1]
%       Y: k X n (10 X 10000)
%           uint8, {0, 1}
%       y: 1 X n (1 X 10000)
%           uint8, {1, 2, ..., 10}
%   valid_data, test_data (ditto)

    [trainX, trainY, trainy] = loadBatch('data_batch_1.mat');
    [validX, validY, validy] = loadBatch('data_batch_2.mat');
    [testX, testY, testy] = loadBatch('test_batch.mat');
    
    % normalize data
    mean_X = mean(trainX, 2);
    std_X = std(trainX, 0, 2);

    trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
    trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);
    validX = validX - repmat(mean_X, [1, size(validX, 2)]);
    validX = validX ./ repmat(std_X, [1, size(validX, 2)]);
    testX = testX - repmat(mean_X, [1, size(testX, 2)]);
    testX = testX ./ repmat(std_X, [1, size(testX, 2)]);
    
    train_data = {trainX, trainY, trainy};
    valid_data = {validX, validY, validy};
    test_data = {testX, testY, testy};
    
end
