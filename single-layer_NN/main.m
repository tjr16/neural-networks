%% Train a multi-linear classifier for cifar-10 dataset
% Dependency: Deep Learning Toolbox
% The path of cifar10 data set and utility functions should be added.

% assume: pwd is the folder where 'main.m' is located
addpath '../cifar-10-batches-mat'  % data set
addpath 'utils' % utilities
% k: #classes, d: dim of feature; n:#images in each .mat
k = 10; d = 3072; n = 10000;
rng(400);
% some options
LOSS_FUNC = 'entropy';      % {'SVM', 'entropy'}
RUN_TEST = false;       % test analytical gradient computation
LARGE_DATASET = false;   % use more training data
DECAY = 1;           % learning rate decay after each epoch
XAVIER = false;          % use Xavier distribution 

%% Important parameters
% note: consider to keep #update instead of #epoch,
% which is n/n_batch * n_epochs

% TODO: set parameters here
% cross entropy
% n_batch=100; eta=1; n_epochs=40; lambda=0;
% n_batch=100; eta=0.001; n_epochs=40; lambda=0;
% n_batch=100; eta=0.001; n_epochs=40; lambda=0.1;	% best
% n_batch=100; eta=0.001; n_epochs=40; lambda=1;
% svm
n_batch=100; eta=0.0001; n_epochs=150; lambda=0.01;
% n_batch=100; eta=0.001; n_epochs=150; lambda=0.01;    % bad
% n_batch=250; eta=0.0001; n_epochs=150; lambda=0.01;

%% Load data
if LARGE_DATASET
    n_valid = 1000;
    n_train = 5 * n - n_valid;
    trainX = zeros(d, n_train);
    trainY = zeros(k, n_train);
    trainy = zeros(1, n_train);
    for i = 0:3
        indices = n*i+1: n*(i+1);
        [tX, tY, ty] = LoadBatch(sprintf('data_batch_%d.mat', i+1));
        trainX(:, indices) = tX;
        trainY(:, indices) = tY;
        trainy(:, indices) = ty;
    end
    [tX, tY, ty] = LoadBatch(sprintf('data_batch_%d.mat', 5));
    trainX(:, n*4+1: end) = tX(:, 1:end-n_valid);
    trainY(:, n*4+1: end) = tY(:, 1:end-n_valid);
    trainy(:, n*4+1: end) = ty(:, 1:end-n_valid);
    validX = tX(:, end-n_valid+1:end);
    validY = tY(:, end-n_valid+1:end);
    validy = ty(:, end-n_valid+1:end); 
else
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [validX, validY, validy] = LoadBatch('data_batch_2.mat');
end

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
if XAVIER
    W = Xavier(k, d);
    b = Xavier(k, 1);
else
    W = 0.01 * randn(k, d);
    b = 0.01 * randn(k, 1);
end

%% Test gradient computation
if RUN_TEST && ~strcmp(LOSS_FUNC, 'SVM')
    runtests('testGradient.m');
end

%% Perform mini-batch GD
GDparams = [n_batch, eta, n_epochs];

[Wstar, bstar, metrics] = MiniBatchGD(...
    trainX_nor, trainY, validX_nor, validY, ...
    GDparams, W, b, lambda, DECAY, strcmp(LOSS_FUNC, 'SVM'));

%% Print accuracy
train_acc = ComputeAccuracy(trainX_nor, trainy, Wstar, bstar, ...
    strcmp(LOSS_FUNC, 'SVM'));
valid_acc = ComputeAccuracy(validX_nor, validy, Wstar, bstar, ...
    strcmp(LOSS_FUNC, 'SVM'));
test_acc = ComputeAccuracy(testX_nor, testy, Wstar, bstar, ...
    strcmp(LOSS_FUNC, 'SVM'));
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
