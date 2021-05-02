%% Multi-layer Perceptron: example
% Dependency: Deep Learning Toolbox

% The path of cifar10 data set and utility functions should be added.
% Suppose 'pwd' is the folder where 'main.m' is located.
addpath '../cifar-10-batches-mat'  % dataset
addpath 'utils' % utilities
rng(400);

%% network parameters
% d: dimension of each layer, from input to output
% first and last dim should be 3072 and 10
global MLP MLP2 MLP3 MLP9
MLP.d = [3072, 10];
MLP2.d = [3072, 50, 10];
MLP3.d = [3072, 50, 50, 10];
MLP9.d = [3072, 50, 30, 20, 20, 10, 10, 10, 10, 10];

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

%% Load Dataset
% A: small, B: middle, C: large
[trainA, validA, testA] = loadData();
[trainB, validB, testB] = loadData(true, 5000);
[trainC, validC, testC] = loadData(true, 1000);

%% pre-experiments：without BN 
% test gradient
for i = 1: 10
    runtests('testGradient.m');
end

% test 3-layer network, test_acc = 53.26%
MLP = MLP3;
[W, b] = initParam();
nn = MultiLayer(W, b); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);

% test 9-layer network, test_acc = 45.58%
MLP = MLP9;
[W, b] = initParam();
nn = MultiLayer(W, b); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);

%% 3-layer with Batch Normalization
% test_acc = 53.40%
MLP = MLP3;
[W, b] = initParam();
nn = MultiLayer(W, b, [], true); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);

% coarse search
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
% good range: 1e-4 ~ 1e-2

% fine search
lam = zeros(1, 10);
lmin = -4; lmax = -2;
for i = 1: 10
    lam(i) = lmin + (lmax - lmin) * rand(1, 1);
end
lam = sort(10.^lam);

tmp = OPT;
figure;	% save validation acc as a figure
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
title('Fine search: lambda'); xlabel('epoch');
ylabel('acc_valid', 'Interpreter', 'none');
OPT = tmp;
% Summary: search range: 5e-4 ~ 3e-2, number of cycles: 2
% best lambda: 0.005932

% train 3 cycles with best lamdba
tmp = OPT;
OPT.n_epoch = 30;
OPT.lambda = 0.005932;
[W, b] = initParam();
nn = MultiLayer(W, b, [], true); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);
OPT = tmp;
% Summary: test_acc = 53.32% (a little bit worse)

% train 3 cycles with best lamdba
tmp = OPT;
OPT.n_epoch = 30;
OPT.lambda = 0.005932;
[W, b] = initParam();
nn = MultiLayer(W, b, [], true); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);
OPT = tmp;
% Summary: test_acc = 53.32% (a little bit worse)

%% 9-layer with Batch Normalization 
% test_acc = 52.26% (big improvement)
MLP = MLP9;
[W, b] = initParam();
nn = MultiLayer(W, b, [], true); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);

% coarse search
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
% good range: 1e-5 ~ 1e-2

% fine search
lam = zeros(1, 10);
lmin = -5; lmax = -2;
for i = 1: 10
    lam(i) = lmin + (lmax - lmin) * rand(1, 1);
end
lam = sort(10.^lam);

tmp = OPT;
figure;	% save validation acc as a figure
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
title('Fine search: lambda'); xlabel('epoch');
ylabel('acc_valid', 'Interpreter', 'none');
OPT = tmp;
% Summary: search range: 5e-4 ~ 3e-2, number of cycles: 2
% best lambda: 0.0015782 (0.0002832 maybe noise)

% train 3 cycles with the same lambda as 3-layer net
tmp = OPT;
OPT.n_epoch = 30;
OPT.lambda = 0.005932;
[W, b] = initParam();
nn = MultiLayer(W, b, [], true); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);
OPT = tmp;
% Summary: test_acc = 52.46% (a little bit better)

%% Sensitivity to initialization

MLP = MLP3;
% generate parameters
sigs = [1e-1, 1e-3, 1e-4];
n_sig = numel(sigs);
Ws = cell(n_sig, 1);
bs = cell(n_sig, 1);
for i = 1: n_sig
    [Ws{i}, bs{i}] = initParam(MLP3, 'Normal', sigs(i));
end

test_acc = zeros(n_sig, 1);
test_acc_bn = zeros(n_sig, 1);

loss_train = cell(n_sig, 1);
loss_valid = cell(n_sig, 1);
loss_train_bn = cell(n_sig, 1);
loss_valid_bn = cell(n_sig, 1);

for i = 1: n_sig
    nn1 = MultiLayer(Ws{i}, bs{i}); nn1 = nn1.train();
    [nn_train1, metrics] = miniBatchGD(trainB, validB, nn1);
    loss_train{i} = metrics(1, :);
    loss_valid{i} = metrics(2, :);
    test_acc(i) = evaluate(nn_train1, testB);
    
    nn2 = MultiLayer(Ws{i}, bs{i}, [], true); nn2 = nn2.train();
    [nn_train2, metrics] = miniBatchGD(trainB, validB, nn2);
    loss_train_bn{i} = metrics(1, :);
    loss_valid_bn{i} = metrics(2, :);
    test_acc_bn(i) = evaluate(nn_train2, testB);
end

% loss plots
% figure;
% for i = 1: n_sig
%     plot(loss_train{i});
%     hold on;
%     plot(loss_valid{i});
% end
% legend('sig=1e-1, train', 'sig=1e-1, valid', ...
%     'sig=1e-3, train', 'sig=1e-3, valid', ...
%     'sig=1e-4, train', 'sig=1e-4, valid');
% title('loss, without BN');
% 
% figure;
% for i = 1: n_sig
%     plot(loss_train_bn{i});
%     hold on;
%     plot(loss_valid_bn{i});
% end
% legend('sig=1e-1, train', 'sig=1e-1, valid', ...
%     'sig=1e-3, train', 'sig=1e-3, valid', ...
%     'sig=1e-4, train', 'sig=1e-4, valid');
% title('loss, with BN');

figure;
for i = 1: n_sig
    plot(loss_valid{i});
    hold on;
end
legend('sig=1e-1', 'sig=1e-3', 'sig=1e-4');
xlabel('epoch'); ylabel('validation acc');
title('loss, without BN');

figure;
for i = 1: n_sig
    plot(loss_valid_bn{i});
    hold on;
end
legend('sig=1e-1', 'sig=1e-3', 'sig=1e-4');
xlabel('epoch'); ylabel('validation acc');
title('loss, with BN');

figure;
semilogx(sigs, test_acc);
hold on;
semilogx(sigs, test_acc_bn);
legend('without BN', 'with BN');
title('test accuracy');

