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

% test 2-layer network, test_acc = 52.8%
% test 3-layer network, test_acc = 52.97%
MLP = MLP3;
[W, b] = initParam();
nn = MultiLayer(W, b); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);

% test 9-layer network, test_acc = 26.79%
MLP = MLP9;
[W, b] = initParam();
nn = MultiLayer(W, b); nn = nn.train();
[nn_train, metrics] = miniBatchGD(trainB, validB, nn);
subplotMetrics(metrics);
evaluate(nn_train, testB);

%% Search best hyper-parameters
if RUN_SEARCH
    [train_data, valid_data, ~] = loadData(true, 5000);
    % --- coarse search ---
    lam = logspace(-5, -1, 8);
    tmp = OPT;
    OPT.n_batch = 100; OPT.lr = 1e-5;
    OPT.cyclic = true; OPT.lr_max = 1e-1;
    OPT.ns = 2 * floor(45000/OPT.n_batch);  % stepsize <-> 2 epochs
    OPT.n_epoch = 8;    % 2 cycles <-> 4 ns <-> 8 epochs
    figure; % save validation acc as a figure
    l_idx = 1;
    for l = lam
        OPT.lambda = l;
        fprintf('%d, lambda=%f\n--------------\n', l_idx, l);
        [~, metrics] = miniBatchGD(train_data, valid_data, nn);
        acc_val = metrics(end, :);
        plot(1: numel(acc_val), acc_val);
        hold on; 
        l_idx = l_idx + 1;
    end
    legend(string(lam));
    title('Coarse search: lambda'); xlabel('epoch'); 
    ylabel('acc_valid', 'Interpreter', 'none');
    OPT = tmp;
    % Summary: search range: 1e-5 ~ 1e-1, number of cycles: 2
    % 3 best: 0.00051795, 0.026827, 0.0019307; good range: 5e-4 ~ 3e-2
    % --- fine search ---
    lam = zeros(1, 10);
    lmin = -4 + log10(5); lmax = -2 + log10(3);
    for i = 1: 10
        lam(i) = lmin + (lmax - lmin) * rand(1, 1);
    end
    lam = sort(10.^lam);
    tmp = OPT;
    OPT.n_batch = 100; OPT.lr = 1e-5;
    OPT.cyclic = true; OPT.lr_max = 1e-1;
    OPT.ns = 2 * floor(45000/OPT.n_batch);	% stepsize <-> 2 epochs
    OPT.n_epoch = 16;	% 4 cycles <-> 8 ns <-> 16 epochs
    figure;	% save validation acc as a figure
    l_idx = 1;
    for l = lam
        OPT.lambda = l;
        fprintf('%d, lambda=%f\n--------------\n', l_idx, l);
        [~, metrics] = miniBatchGD(train_data, valid_data, nn);
        acc_val = metrics(end, :);
        plot(1: numel(acc_val), acc_val);
        hold on; 
        l_idx = l_idx + 1;
    end
    legend(string(lam));
    title('Fine search: lambda'); xlabel('epoch');
    ylabel('acc_valid', 'Interpreter', 'none');
    OPT = tmp;
    % Summary: search range: 5e-4 ~ 3e-2, number of cycles: 4
    % 3 best: 0.0021731, 0.0036329, 0.0011671; good range: 5e-4 ~ 3e-2
end

%% final test
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = OPT;
OPT.n_batch = 100; OPT.lr = 1e-5;
OPT.cyclic = true; OPT.lr_max = 1e-1;
OPT.ns = 2 * floor(49000/OPT.n_batch);	% stepsize <-> 2 epochs
OPT.n_epoch = 12;	% 3 cycles <-> 6 ns <-> 12 epochs
OPT.lambda = 0.0021731;	% best lr

[nn_trained, metrics] = miniBatchGD(train_data, valid_data, nn);
figure;
subplotMetrics(metrics);

nn_eval = nn_trained.eval();
nn_final = nn_eval.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);

OPT = tmp;
% Summary: test accuracy, 52.83%


%% performance optimization

%% 1. add more hidden nodes
tmpNN = NN2; tmpGD = OPT;
NN2.m = 100;    % TODO: set #nodes here
OPT.n_batch = 100; OPT.lr = 1e-5;
OPT.cyclic = true; OPT.lr_max = 1e-1;
OPT.ns = 2 * floor(45000/OPT.n_batch);  % stepsize <-> 2 epochs
OPT.n_epoch = 8;    % 2 cycles <-> 4 ns <-> 8 epochs
% init network
[W1, b1] = initParam();
nn1 = DoubleLayer(W1, b1);
% search for proper regularization
[train_data, valid_data, ~] = loadData(true, 5000);
lam = logspace(-5, -1, 8);
figure; % save validation acc as a figure
l_idx = 1;
for l = lam
    OPT.lambda = l;
    fprintf('%d, lambda=%f\n--------------\n', l_idx, l);
    [~, metrics] = miniBatchGD(train_data, valid_data, nn1);
    acc_val = metrics(end, :);
    plot(1: numel(acc_val), acc_val);
    hold on;
    l_idx = l_idx + 1;
end
legend(string(lam));
title('Search for lambda: more hidden nodes'); xlabel('epoch'); 
ylabel('acc_valid', 'Interpreter', 'none');

% good lr value: 3.7276e-5 (100 nodes)
%                1.9307e-3 (150 nodes)

% final test
OPT.ns = 2 * floor(49000/OPT.n_batch);	% stepsize <-> 2 epochs
OPT.n_epoch = 12;	% 3 cycles <-> 6 ns <-> 12 epochs
OPT.lambda = 3.7276e-5;  % TODO: set best lr here
[train_data, valid_data, test_data] = loadData(true, 1000);
[nn_trained1, metrics] = miniBatchGD(train_data, valid_data, nn1);
figure;
plotMetrics(metrics);
nn_final = nn_trained1.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);
NN2 = tmpNN; OPT = tmpGD;

% Summary: test accuracy, 53.05% (100 nodes)
%                         54.84% (150 nodes)

%% 2. ensemble learning for several networks
n_model = 3;
nns = cell(1, n_model);   % NN cell
Ws = cell(4, n_model);    % param cells
bs = cell(4, n_model);
metrics = cell(1, n_model);   % metrics cell
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = OPT;
OPT.n_batch = 100; OPT.lr = 1e-5;
OPT.cyclic = true; OPT.lr_max = 1e-1;
OPT.ns = 2 * floor(49000/OPT.n_batch);	% stepsize <-> 2 epochs
OPT.n_epoch = 4;	% 1 cycle = 4 epochs
OPT.lambda = 0.0021731;	% best lr

for i = 1:n_model	% each model
    fprintf('Model: %d\n', i);
    [Ws{1, i}, b{1, i}] = initParam();
    nns{i} = DoubleLayer(Ws{1, i}, b{1, i});
    for j = 1:3 % each cycle: save model parameters
        [nns{i}, metrics_] = miniBatchGD(train_data, valid_data, nns{i});
        metrics{i} = [metrics{i}, metrics_];
        Ws{j+1, i} = nns{i}.W;
        bs{j+1, i} = nns{i}.b;        
    end
end
% plot metrics of each model
for i = 1:n_model
    figure(i*1000); % distinguish different figures
	subplotMetrics(metrics{i});
end
% ensemble
accs = ensemble(nns, test_data);
for i = 1: n_model
    fprintf("Test accuracy, model %d: %f\n", i, accs{i});
end
fprintf("Test accuracy, ensemble model: %f\n", accs{end});
OPT = tmp;
% Summary:
% Test accuracy, model 1: 0.523900
% Test accuracy, model 2: 0.523300
% Test accuracy, model 3: 0.527400
% Test accuracy, ensemble model: 0.532600

%% 3. ensemble learning for several cycles
n_cycle = 10;
nns = cell(1, n_cycle+1);   % NN cell
metrics = cell(1, n_cycle);   % metrics cell
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = OPT;
OPT.n_batch = 100; OPT.lr = 1e-5;
OPT.cyclic = true; OPT.lr_max = 1e-1;
OPT.ns = 2 * floor(49000/OPT.n_batch);	% stepsize <-> 2 epochs
OPT.n_epoch = 4;	% 1 cycle = 4 epochs
OPT.lambda = 0.0021731;	% best lr

all_metrics = zeros(6, n_cycle * OPT.n_epoch);
[W, b] = initParam();
nns{1} = DoubleLayer(W, b);

% each cycle: save model
for i = 1:n_cycle 
    [nns{i+1}, metrics{i}] = miniBatchGD(train_data, valid_data, nns{i});
    all_metrics(:, 1 + OPT.n_epoch * (i-1): OPT.n_epoch * i) = metrics{i};
end

% plot metrics
subplotMetrics(all_metrics);

% ensemble
accs = ensemble(nns(2:end), test_data);
for i = 1: n_cycle 
    fprintf("Test accuracy, model %d: %f\n", i, accs{i});
end
fprintf("Test accuracy, ensemble model: %f\n", accs{end});
OPT = tmp;

%% 4. dropout
dropout = 0.5;
[train_data, valid_data, test_data] = loadData(true, 1000);

% config
tmpNN = NN2; tmpGD = OPT;
NN2.m = 100;    % TODO: set #nodes here
OPT.n_batch = 100; OPT.lr = 1e-5;
OPT.cyclic = true; OPT.lr_max = 1e-1;
OPT.ns = 2 * floor(49000/OPT.n_batch);  % stepsize <-> 2 epochs
OPT.n_epoch = 40;    % 4 epochs for 1 cycle
OPT.lambda = 0;

% init network
[W, b] = initParam();
nn = DoubleLayer(W, b, dropout);

% train
[nn_trained, metrics] = miniBatchGD(train_data, valid_data, nn);

% evaluate
figure;
subplotMetrics(metrics);

nn_eval = nn_trained.eval();
[nn_final, ~] = nn_eval.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);

NN2 = tmpNN; OPT = tmpGD;

% 10 cycles: 51.77%

%% 5.learning rate range test
[train_data, valid_data, test_data] = loadData(true, 5000);
% stepsize = 8 epochs = 8 * 450 iterations
tmp = OPT;
OPT.n_batch = 100; 
OPT.cyclic = true; 
OPT.n_epoch = 8;
OPT.ns = OPT.n_epoch * floor(45000/OPT.n_batch);  % ns == #update_steps
OPT.lambda = 0.0021731;	% best lr

% lr range test
OPT.lr = 1e-9; OPT.lr_max = 1e0;
[W, b] = initParam();
nn = DoubleLayer(W, b);
% really time consuming
[etas, acc, acc_val] = lrRangeTest(train_data, valid_data, nn);

% log plot
figure;
semilogx(etas,acc,etas,acc_val)
grid on
xlabel('log(learning rate)');
ylabel('accuracy');
legend('training', 'validation');

% log plot, only valid
figure;
semilogx(etas,acc_val)
grid on
xlabel('log(learning rate)');
ylabel('valid accuracy');

% linear plot
figure;
plot(etas, acc_val);
xlabel('eta');
ylabel('validation accuracy');

OPT = tmp;

%% final test for 5.
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = OPT;
OPT.n_batch = 100; OPT.lr = 5e-4;
OPT.cyclic = true; OPT.lr_max = 2e-2;
OPT.ns = 2 * floor(49000/OPT.n_batch);	% stepsize <-> 2 epochs
OPT.n_epoch = 20;	% 3 cycles <-> 6 ns <-> 12 epochs
OPT.lambda = 0.0021731;	% best lr

[nn_trained, metrics] = miniBatchGD(train_data, valid_data, nn);
figure;
subplotMetrics(metrics);

nn_eval = nn_trained.eval();
nn_final = nn_eval.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);

OPT = tmp;
