%% Train a double-layer NN on cifar-10 dataset
% Dependency: Deep Learning Toolbox

% The path of cifar10 data set and utility functions should be added.
% Suppose 'pwd' is the folder where 'main.m' is located.
addpath '../cifar-10-batches-mat'  % dataset
addpath 'utils' % utilities
rng(400);
RUN_TEST = false;

%% global struct with parameters of double-layer NN
% k: #classes (output nodes), m: #hidden nodes, d: #features (input nodes)
global NN2
NN2.k = 10;
NN2.m = 50;
NN2.d = 3072;

%% global struct with parameters of mini-batch GD
% n_batch: batch size, 
% lr: learning rate (it is minimum lr if cyclic),
% n_epoch: #epochs, lambda: L2 penalty coefficient
% cyclic: use cyclic lr, lr_max: maximum learning rate
% ns: stepsize = k*floor(n/n_batch), k<-[2,8]
% [NOTE] consider to keep #update steps as a constant
global GD2
GD2.n_batch = 100;
GD2.lr = 1e-5;   
GD2.n_epoch = 40;
GD2.lambda = 0;
% cyclic lr parameters
GD2.cyclic = true;
GD2.lr_max = 1e-1;
GD2.ns = 500;

%% Initialize network parameters
[W, b] = initParam();
nn = DoubleLayer(W, b);

%% Run some tests
if RUN_TEST
    [train_data, valid_data, test_data] = loadData();   % use small dataset
    %% automatically test gradient computation
    runtests('testGradient.m');
    %% test GD algorithm
    tmp = GD2;
    GD2.n_epoch = 500; GD2.lr = 1e-3;
    nn0 = DoubleLayer(W, b);
    train_small = cell(1,3); valid_small = cell(1,3);
    for i = 1:2
        train_small{i} = train_data{i}(:, 1:100);
        valid_small{i} = valid_data{i}(:, 1:100);
    end
    train_small{3} = train_data{3}(1:100);
    valid_small{3} = valid_data{3}(1:100);
    [~, metrics] = miniBatchGD(train_small, valid_small, nn0);
    plotMetrics(metrics);
    GD2 = tmp;
    %% test cyclic learning rate in one cycle
    tmp = GD2;
    GD2.n_batch = 100; GD2.lr = 1e-5; GD2.n_epoch = 10;
    GD2.lambda = 0.01; GD2.cyclic = true; GD2.lr_max = 1e-1; GD2.ns = 500;
    nn0 = DoubleLayer(W, b);
    [~, metrics] = miniBatchGD(train_data, valid_data, nn0);
    plotMetrics(metrics);
    GD2 = tmp;
    %% test cyclic learning rate in three cycles
    tmp = GD2;
    GD2.n_batch = 100; GD2.lr = 1e-5; GD2.n_epoch = 48;
    GD2.lambda = 0.01; GD2.cyclic = true; GD2.lr_max = 1e-1; GD2.ns = 800;
    nn0 = DoubleLayer(W, b);
    [~, metrics] = miniBatchGD(train_data, valid_data, nn0);
    plotMetrics(metrics);
    GD2 = tmp;
end

%% Coarse search
[train_data, valid_data, test_data] = loadData(true, 5000);

%% Fine search

%% Final
[train_data, valid_data, test_data] = loadData(true, 1000);

%% Perform mini-batch GD
[nn_trained, metrics] = miniBatchGD(train_data, valid_data, nn);
plotMetrics(metrics);



%% Print accuracy
% fprintf("Accuracy on training set: %f\n", train_acc);
% fprintf("Accuracy on validation set: %f\n", valid_acc);
% fprintf("Accuracy on test set: %f\n", test_acc);
