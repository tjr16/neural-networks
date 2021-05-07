%% Multi-layer Perceptron: performance improvement
% Dependency: Deep Learning Toolbox

% The path of cifar10 data set and utility functions should be added.
% Suppose 'pwd' is the folder where 'main.m' is located.
addpath '../cifar-10-batches-mat'  % dataset
addpath 'utils' % utilities
rng(400);

%% network parameters
% d: dimension of each layer, from input to output
% first and last dim should be 3072 and 10
global MLP 
% global MLP2 MLP3 MLP9
MLP.d = [3072, 10];
MLP6.d = [3072 200 200 200 200 200 10];

MLP_list = cell(8, 1);  % 2-layer to 9-layer
for i = 1: 8
    MLP_list{i}.d = [3072, 50 * ones(1, i-1), 10];
end

global BN
BN.alpha = 0.7;

%% optimization parameters
% n_batch: batch size, 
% lr: learning rate (it is minimum lr if cyclic),
% n_epoch: #epochs, lambda: L2 penalty coefficient
% cyclic: use cyclic lr, lr_max: maximum learning rate
% ns: stepsize = k*floor(n/n_batch), k<-[2,8]
global OPT
OPT.n_batch = 100;
OPT.lr = 1e-5;   
OPT.n_epoch = 20;
OPT.lambda = 0.005;
% cyclic lr parameters
OPT.cyclic = true;
OPT.lr_max = 1e-1;
OPT.ns = 5 * 45000 / OPT.n_batch;	% 1 cycle = 2 * 5 epoch

%% load dataset
% A: small, B: middle, C: large
[trainA, validA, testA] = loadData();
[trainB, validB, testB] = loadData(true, 5000);
[trainC, validC, testC] = loadData(true, 1000);

%% network architecture
% test_acc = zeros(numel(MLP_list), 1);
% for i = 1: numel(MLP_list)
%     fprintf('%d-layer network\n--------------\n', i+1);
%     MLP = MLP_list{i};
%     [W, b] = initParam();
%     nn = MultiLayer(W, b, [], true);
%     [nn_train, metrics] = miniBatchGD(trainB, validB, nn);
%     subplotMetrics(metrics, [], [], sprintf('%d-layer network', i+1));
%     test_acc(i) = evaluate(nn_train, testB);
% end
% 
% figure('Name', 'Test accuracy: #layers');
% plot(2: numel(MLP_list)+ 1, test_acc, 'o-');
% grid on;
% title('test accuracy');

% fine search: #layer
valid_acc = zeros(numel(MLP_list) - 3, 1);
OPT.n_epoch = 40;
for i = 4: numel(MLP_list)
    fprintf('%d-layer network\n--------------\n', i+1);
    MLP = MLP_list{i};
    [W, b] = initParam();
    nn = MultiLayer(W, b, [], true);
    [nn_train, metrics] = miniBatchGD(trainB, validB, nn);
    subplotMetrics(metrics, [], [], sprintf('%d-layer network', i+1));
    valid_acc(i-3) = evaluate(nn_train, validB);
end
figure('Name', 'Valid accuracy: #layers');
plot(5: numel(MLP_list)+ 1, valid_acc, 'o-');
grid on;
title('valid accuracy');

% search: #nodes
% I use a 6-layer network
MLP_list = cell(4, 1);  % 50, 100, 150, 200 nodes
valid_acc = zeros(4, 1);
for i = 1: 4
    MLP_list{i}.d = [3072, 50 * i * ones(1, 5), 10];
end

for i = 1: 4
    fprintf('%d nodes\n--------------\n', 50 * i);
    MLP = MLP_list{i};
    [W, b] = initParam();
    nn = MultiLayer(W, b, [], true);
    [nn_train, metrics] = miniBatchGD(trainB, validB, nn);
    subplotMetrics(metrics, [], [], sprintf('6-layer network, %d nodes', 50 * i));
    valid_acc(i) = evaluate(nn_train, validB);
end
figure('Name', 'Valid accuracy: #nodes');
plot(50:50:200, valid_acc, 'o-');
grid on;
title('valid accuracy');

% I use 200 nodes
evaluate(nn_train, testB);

%% lambda search

MLP = MLP6;
% [W, b] = initParam();
load('weights/MLP6_init.mat');
nn = MultiLayer(W, b, [], true);

% coarse search
OPT.n_epoch = 20;
lam = logspace(-7, 0, 8);
tmp = OPT;
figure; % save validation acc as a figure
l_idx = 1;
for l = lam
    OPT.lambda = l;
    fprintf('%d, lambda=%f\n--------------\n', l_idx, l);
    [~, metrics] = miniBatchGD(trainB, validB, nn);
    acc_val = metrics(end, :);
    plot(1: numel(acc_val), acc_val);
    hold on; 
    l_idx = l_idx + 1;
end
legend(string(lam));
title('Coarse search: lambda'); xlabel('epoch'); 
ylabel('acc_valid', 'Interpreter', 'none');
OPT = tmp;
% Summary: search range: 1e-7 ~ 1e0, number of cycles: 2
% good range: 1e-1 ~ 1e-3

% fine search
n_fine = 5;
lam = zeros(1, n_fine);
lmin = -3; lmax = -1;
for i = 1: n_fine
    lam(i) = lmin + (lmax - lmin) * rand(1, 1);
end
lam = sort(10.^lam);

tmp = OPT;
OPT.n_epoch = 30;
acc_val_list = zeros(n_fine, 3);

for j = 1: 3
    figure;	% save validation acc as a figure
    for i = 1: n_fine
        l = lam(i);
        OPT.lambda = l;
        fprintf('%d, lambda=%f\n--------------\n', i, l);
        [~, metrics] = miniBatchGD(trainB, validB, nn);
        acc_val = metrics(end, :);
        acc_val_list(i, j) = acc_val(end);
        plot(1: numel(acc_val), acc_val);
        hold on; 
    end
    legend(string(lam));
    title(sprintf('Fine search %d: lambda', j)); xlabel('epoch');
    ylabel('acc_valid', 'Interpreter', 'none');
end

OPT = tmp;
figure;
plot(lam, mean(acc_val_list, 2), '-o');
title('fine search');
xlabel('lambda');
ylabel('mean validation acc');
disp(lam);  % random lambda
disp(mean(acc_val_list, 2)');  % mean validation acc
% Summary: search range: 1e-3 ~ 1e-1, number of cycles: 3
% best lambda: 0.0079024 (valid_acc= 59.17%)

% final test: train 3 cycles with best lamdba
tmp = OPT;
OPT.n_epoch = 30;
% OPT.lambda = 0.0079024;
OPT.lambda = 0.005;
load('weights/MLP6_init.mat');
nn = MultiLayer(W, b, [], true);
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);
OPT = tmp;
% Summary: test_acc = 0.575900 (use same W, lambda=0.005: 0.576900)
W = nn_train.W; b = nn_train.b;
save('weights/MLP6_fine.mat', 'W', 'b');


%% data augmentation (geometric and photometric jitter)
load('weights/MLP6_fine.mat');
MLP = MLP6;
OPT.n_epoch = 40; OPT.lambda = 0.0079024;
nn = MultiLayer(W, b, [], true);
[nn_train, metrics] = miniBatchGD(trainB, validB, nn, true);
subplotMetrics(metrics);
evaluate(nn_train, testB);

%% dropout
MLP = MLP6;
OPT.n_epoch = 40; OPT.lambda = 0.0079024;

dropout_list = [0.1, 0.3, 0.5];
test_acc = zeros(3, 1);
for i = 1: 3
    dropout = dropout_list(i);
    nn = MultiLayer(W, b, dropout, true);
    [nn_train, metrics] = miniBatchGD(trainB, validB, nn);
    subplotMetrics(metrics, [], [], sprintf('dropout = %f', dropout));
    test_acc(i) = evaluate(nn_train, testB);
end
% 200 nodes, 0.1 dropout: 0.5710 0.5646
% 200 nodes, 0.3 dropout: 0.5244 0.517100

MLP.d = [3072 300 300 300 300 300 10];
[W, b] = initParam();
nn = MultiLayer(W, b, 1/3, true);
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics, [], [], sprintf('dropout = %f', dropout));
test_acc(i) = evaluate(nn_train, testB);
