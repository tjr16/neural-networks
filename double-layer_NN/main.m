%% Train a double-layer NN on cifar-10 dataset
% Dependency: Deep Learning Toolbox

% The path of cifar10 data set and utility functions should be added.
% Suppose 'pwd' is the folder where 'main.m' is located.
addpath '../cifar-10-batches-mat'  % dataset
addpath 'utils' % utilities
rng(400);
RUN_TEST = false;   % test implementation
RUN_SEARCH = false; % search hyper-parameters

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

% --- at least, all code above should be run ---
%% Initialize network parameters
[W, b] = initParam();
nn = DoubleLayer(W, b);

%% Run some tests
if RUN_TEST
    [train_data, valid_data, test_data] = loadData();   % use small dataset
    % --- automatically test gradient computation ---
    runtests('testGradient.m');
    % --- test GD algorithm ---
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
    % --- test cyclic learning rate in one cycle ---
    tmp = GD2;
    GD2.n_batch = 100; GD2.lr = 1e-5; GD2.n_epoch = 10;
    GD2.lambda = 0.01; GD2.cyclic = true; GD2.lr_max = 1e-1; GD2.ns = 500;
    nn0 = DoubleLayer(W, b);
    [~, metrics] = miniBatchGD(train_data, valid_data, nn0);
    plotMetrics(metrics);
    GD2 = tmp;
    % --- test cyclic learning rate in three cycles ---
    tmp = GD2;
    GD2.n_batch = 100; GD2.lr = 1e-5; GD2.n_epoch = 48;
    GD2.lambda = 0.01; GD2.cyclic = true; GD2.lr_max = 1e-1; GD2.ns = 800;
    nn0 = DoubleLayer(W, b);
    [~, metrics] = miniBatchGD(train_data, valid_data, nn0);
    plotMetrics(metrics);
    GD2 = tmp;
end

%% Search best hyper-parameters
if RUN_SEARCH
    [train_data, valid_data, ~] = loadData(true, 5000);
    % --- coarse search ---
    lam = logspace(-5, -1, 8);
    tmp = GD2;
    GD2.n_batch = 100; GD2.lr = 1e-5;
    GD2.cyclic = true; GD2.lr_max = 1e-1;
    GD2.ns = 2 * floor(45000/GD2.n_batch);  % stepsize <-> 2 epochs
    GD2.n_epoch = 8;    % 2 cycles <-> 4 ns <-> 8 epochs
    figure; % save validation acc as a figure
    l_idx = 1;
    for l = lam
        GD2.lambda = l;
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
    GD2 = tmp;
    % Summary: search range: 1e-5 ~ 1e-1, number of cycles: 2
    % 3 best: 0.00051795, 0.026827, 0.0019307; good range: 5e-4 ~ 3e-2
    % --- fine search ---
    lam = zeros(1, 10);
    lmin = -4 + log10(5); lmax = -2 + log10(3);
    for i = 1: 10
        lam(i) = lmin + (lmax - lmin) * rand(1, 1);
    end
    lam = sort(10.^lam);
    tmp = GD2;
    GD2.n_batch = 100; GD2.lr = 1e-5;
    GD2.cyclic = true; GD2.lr_max = 1e-1;
    GD2.ns = 2 * floor(45000/GD2.n_batch);	% stepsize <-> 2 epochs
    GD2.n_epoch = 16;	% 4 cycles <-> 8 ns <-> 16 epochs
    figure;	% save validation acc as a figure
    l_idx = 1;
    for l = lam
        GD2.lambda = l;
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
    GD2 = tmp;
    % Summary: search range: 5e-4 ~ 3e-2, number of cycles: 4
    % 3 best: 0.0021731, 0.0036329, 0.0011671; good range: 5e-4 ~ 3e-2
end

%% final test
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = GD2;
GD2.n_batch = 100; GD2.lr = 1e-5;
GD2.cyclic = true; GD2.lr_max = 1e-1;
GD2.ns = 2 * floor(49000/GD2.n_batch);	% stepsize <-> 2 epochs
GD2.n_epoch = 12;	% 3 cycles <-> 6 ns <-> 12 epochs
GD2.lambda = 0.0021731;	% best lr

[nn_trained, metrics] = miniBatchGD(train_data, valid_data, nn);
figure;
subplotMetrics(metrics);

nn_eval = nn_trained.eval();
nn_final = nn_eval.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);

GD2 = tmp;
% Summary: test accuracy, 52.83%


%% performance optimization

%% 1. add more hidden nodes
tmpNN = NN2; tmpGD = GD2;
NN2.m = 100;    % TODO: set #nodes here
GD2.n_batch = 100; GD2.lr = 1e-5;
GD2.cyclic = true; GD2.lr_max = 1e-1;
GD2.ns = 2 * floor(45000/GD2.n_batch);  % stepsize <-> 2 epochs
GD2.n_epoch = 8;    % 2 cycles <-> 4 ns <-> 8 epochs
% init network
[W1, b1] = initParam();
nn1 = DoubleLayer(W1, b1);
% search for proper regularization
[train_data, valid_data, ~] = loadData(true, 5000);
lam = logspace(-5, -1, 8);
figure; % save validation acc as a figure
l_idx = 1;
for l = lam
    GD2.lambda = l;
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
GD2.ns = 2 * floor(49000/GD2.n_batch);	% stepsize <-> 2 epochs
GD2.n_epoch = 12;	% 3 cycles <-> 6 ns <-> 12 epochs
GD2.lambda = 3.7276e-5;  % TODO: set best lr here
[train_data, valid_data, test_data] = loadData(true, 1000);
[nn_trained1, metrics] = miniBatchGD(train_data, valid_data, nn1);
figure;
plotMetrics(metrics);
nn_final = nn_trained1.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);
NN2 = tmpNN; GD2 = tmpGD;

% Summary: test accuracy, 53.05% (100 nodes)
%                         54.84% (150 nodes)

%% 2. ensemble learning for several networks
n_model = 3;
nns = cell(1, n_model);   % NN cell
Ws = cell(4, n_model);    % param cells
bs = cell(4, n_model);
metrics = cell(1, n_model);   % metrics cell
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = GD2;
GD2.n_batch = 100; GD2.lr = 1e-5;
GD2.cyclic = true; GD2.lr_max = 1e-1;
GD2.ns = 2 * floor(49000/GD2.n_batch);	% stepsize <-> 2 epochs
GD2.n_epoch = 4;	% 1 cycle = 4 epochs
GD2.lambda = 0.0021731;	% best lr

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
GD2 = tmp;
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
tmp = GD2;
GD2.n_batch = 100; GD2.lr = 1e-5;
GD2.cyclic = true; GD2.lr_max = 1e-1;
GD2.ns = 2 * floor(49000/GD2.n_batch);	% stepsize <-> 2 epochs
GD2.n_epoch = 4;	% 1 cycle = 4 epochs
GD2.lambda = 0.0021731;	% best lr

all_metrics = zeros(6, n_cycle * GD2.n_epoch);
[W, b] = initParam();
nns{1} = DoubleLayer(W, b);

% each cycle: save model
for i = 1:n_cycle 
    [nns{i+1}, metrics{i}] = miniBatchGD(train_data, valid_data, nns{i});
    all_metrics(:, 1 + GD2.n_epoch * (i-1): GD2.n_epoch * i) = metrics{i};
end

% plot metrics
subplotMetrics(all_metrics);

% ensemble
accs = ensemble(nns(2:end), test_data);
for i = 1: n_cycle 
    fprintf("Test accuracy, model %d: %f\n", i, accs{i});
end
fprintf("Test accuracy, ensemble model: %f\n", accs{end});
GD2 = tmp;

%% 4. dropout
dropout = 0.5;
[train_data, valid_data, test_data] = loadData(true, 1000);

% config
tmpNN = NN2; tmpGD = GD2;
NN2.m = 100;    % TODO: set #nodes here
GD2.n_batch = 100; GD2.lr = 1e-5;
GD2.cyclic = true; GD2.lr_max = 1e-1;
GD2.ns = 2 * floor(49000/GD2.n_batch);  % stepsize <-> 2 epochs
GD2.n_epoch = 40;    % 4 epochs for 1 cycle
GD2.lambda = 0;

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

NN2 = tmpNN; GD2 = tmpGD;

% 10 cycles: 51.77%

%% 5.learning rate range test
[train_data, valid_data, test_data] = loadData(true, 5000);
% stepsize = 8 epochs = 8 * 450 iterations
tmp = GD2;
GD2.n_batch = 100; 
GD2.cyclic = true; 
GD2.n_epoch = 8;
GD2.ns = GD2.n_epoch * floor(45000/GD2.n_batch);  % ns == #update_steps
GD2.lambda = 0.0021731;	% best lr

% lr range test
GD2.lr = 1e-9; GD2.lr_max = 1e0;
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

GD2 = tmp;

%% final test for 5.
[train_data, valid_data, test_data] = loadData(true, 1000);
tmp = GD2;
GD2.n_batch = 100; GD2.lr = 5e-4;
GD2.cyclic = true; GD2.lr_max = 2e-2;
GD2.ns = 2 * floor(49000/GD2.n_batch);	% stepsize <-> 2 epochs
GD2.n_epoch = 20;	% 3 cycles <-> 6 ns <-> 12 epochs
GD2.lambda = 0.0021731;	% best lr

[nn_trained, metrics] = miniBatchGD(train_data, valid_data, nn);
figure;
subplotMetrics(metrics);

nn_eval = nn_trained.eval();
nn_final = nn_eval.forward(test_data{1});
P_final = nn_final.output();
acc_final = computeAccuracy(P_final, test_data{3});
fprintf("Accuracy on test set: %f\n", acc_final);

GD2 = tmp;
