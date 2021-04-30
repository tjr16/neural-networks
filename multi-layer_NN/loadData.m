function [train_data, valid_data, test_data] = loadData(large, valid_size)
% A function that loads and preprocesses training, validation
% and test datasets.
% ----------
% Argument:
%   large: bool, using all data or not
%   valid_size: the amount of validation data when using large dataset
%       integer, 1 <= valid_size <= 10000
% Return:
%   train_data: cell, 1 X 3
%       X: image pixel data, d X n (3072 X 10000) 
%           double, [0, 1]
%       Y: k X n (10 X 10000)
%           uint8, {0, 1}
%       y: 1 X n (1 X 10000)
%           uint8, {1, 2, ..., 10}
%   valid_data, test_data (ditto)

    if nargin < 1
        large = false;
    end
    
    if nargin < 2
        valid_size = 5000;
    end
    if (valid_size < 1) || (valid_size > 10000)
        error('Invalid valid_size argument!');
    end
    
    global MLP
    if large  % #training data >= 40000
        train_size = 50000-valid_size;
        trainX = zeros(MLP.d(1), train_size);
        trainY = zeros(MLP.d(end), train_size);
        trainy = zeros(1, train_size);
        for i = 1: 4
            [loadX, loadY, loady] = loadBatch(sprintf('data_batch_%d.mat', i));
            trainX(:, (i-1)*10000+1:i*10000) = loadX;
            trainY(:, (i-1)*10000+1:i*10000) = loadY;
            trainy(:, (i-1)*10000+1:i*10000) = loady;
        end
        [loadX, loadY, loady] = loadBatch('data_batch_5.mat');
        trainX(:, 40001:train_size) = loadX(:, 1:10000-valid_size);
        trainY(:, 40001:train_size) = loadY(:, 1:10000-valid_size);
        trainy(:, 40001:train_size) = loady(:, 1:10000-valid_size);
        validX = loadX(:, end-valid_size+1:end);
        validY = loadY(:, end-valid_size+1:end);
        validy = loady(:, end-valid_size+1:end);
    else  % #training data == 10000
        [trainX, trainY, trainy] = loadBatch('data_batch_1.mat');
        [validX, validY, validy] = loadBatch('data_batch_2.mat');  
    end
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
