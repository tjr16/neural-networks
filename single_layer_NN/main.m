%% Train a multi-linear classifier for cifar-10 dataset
% Dependency: Deep Learning Toolbox
% The path of cifar10 data set and utility functions should be added.

addpath '../cifar-10-batches-mat'  % data set
addpath 'utils' % utilities
k = 10; d = 3072;
rng(400);
DEBUG = false;

%% Load data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validX, validY, validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

%% Preprocess data
train_mean_X = mean(trainX, 2);
train_std_X = std(trainX, 0, 2);
% normalized data based on the distribution of the training set
trainX_nor = trainX - repmat(train_mean_X, [1, size(trainX, 2)]);
trainX_nor = trainX_nor ./ repmat(train_std_X, [1, size(trainX_nor, 2)]);
validX_nor = validX - repmat(train_mean_X, [1, size(validX, 2)]);
validX_nor = validX_nor ./ repmat(train_std_X, [1, size(validX_nor, 2)]);
testX_nor = testX - repmat(train_mean_X, [1, size(testX, 2)]);
testX_nor = testX_nor ./ repmat(train_std_X, [1, size(testX_nor, 2)]);

%% Initialize parameters
W = 0.01 * randn(k, d);
b = 0.01 * randn(k, 1);

%% Test gradient computation
if DEBUG
    runtests('testGradient.m');
end

%% Perform mini-batch GD
% note: consider to keep #update instead of #epoch,
% which is n/n_batch * n_epochs

% TODO: set parameters here
n_batch=100; eta=1; n_epochs=40; lambda=0;
% n_batch=100; eta=0.001; n_epochs=40; lambda=0;
% n_batch=100; eta=0.001; n_epochs=40; lambda=0.1;
% n_batch=100; eta=0.001; n_epochs=40; lambda=1;

GDparams = [n_batch, eta, n_epochs];

[Wstar, bstar, metrics] = MiniBatchGD(...
    trainX_nor, trainY, validX_nor, validY, GDparams, W, b, lambda);

%% Print accuracy
train_acc = ComputeAccuracy(trainX_nor, trainy, Wstar, bstar);
valid_acc = ComputeAccuracy(validX_nor, validy, Wstar, bstar);
test_acc = ComputeAccuracy(testX_nor, testy, Wstar, bstar);
fprintf("Accuracy on training set: %f\n", train_acc);
fprintf("Accuracy on validation set: %f\n", valid_acc);
fprintf("Accuracy on test set: %f\n", test_acc);

%% Plot loss and cost figures
loss_train = metrics(1, :); loss_valid = metrics(2, :);
cost_train = metrics(3, :); cost_valid = metrics(4, :);

figure(1);
plot(loss_train, '-bo');
hold on;
plot(loss_valid, '-rx');
legend('training', 'validation');
xlabel('epoch'); ylabel('loss'); title('Mean Loss');

figure(2);
plot(cost_train, '-bo');
hold on;
plot(cost_valid, '-rx');
legend('training', 'validation');
xlabel('epoch'); ylabel('cost'); title('Cost');

%% Display weights
s_im = cell(10, 1);
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(3);
montage(s_im, 'Size', [2,5]);
