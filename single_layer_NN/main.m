% Exercise 1: Training a multi-linear classifier
% Dependency: Deep Learning Toolbox
% The path of cifar10 data set and utility functions should be added.

k = 10; d = 3072;
% `n` value in the functions depends on the batch size
addpath 'cifar-10-batches-mat'  % data set
addpath 'utils' % utilities

%% 1. Load data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validX, validY, validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

%% 2. Preprocess data
train_mean_X = mean(trainX, 2);
train_std_X = std(trainX, 0, 2);
% normalized
trainX_nor = trainX - repmat(train_mean_X, [1, size(trainX, 2)]);
trainX_nor = trainX_nor ./ repmat(train_std_X, [1, size(trainX_nor, 2)]);

valid_mean_X = mean(validX, 2);
valid_std_X = std(validX, 0, 2);
validX_nor = validX - repmat(valid_mean_X, [1, size(validX, 2)]);
validX_nor = validX_nor ./ repmat(valid_std_X, [1, size(validX_nor, 2)]);

test_mean_X = mean(testX, 2);
test_std_X = std(testX, 0, 2);
testX_nor = testX - repmat(test_mean_X, [1, size(testX, 2)]);
testX_nor = testX_nor ./ repmat(test_std_X, [1, size(testX_nor, 2)]);

%% 3. Initialize parameters
W = 0.01 * randn(k, d);
b = 0.01 * randn(k, 1);

%% Some components
% 4. Build classifier
% P = EvaluateClassifier(trainX(:, 1:100), W, b);
% 5. Compute cost
% 6. Compute accuracy
% 7. Compute gradient
runtests('testGradient.m');

%% 8.Perform mini-batch GD
% I = reshape(A.data', 32, 32, 3, 10000);
% I = permute(I, [2, 1, 3, 4]);
% montage(I(:, :, :, 1:500), 'Size', [5,5]);
