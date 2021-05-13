%% Recurrent Neural Network: example
% Dependency: Deep Learning Toolbox

% The path of goblet-book data set and utility functions should be added.
% Suppose 'pwd' is the folder where 'main.m' is located.
addpath '../goblet-book'  % dataset
addpath 'utils' % utilities
rng(400);

%% information of book dataset
global INFO
INFO = loadData();

% notations
% h_t = tanh(W * h_t-1 + U * X_t + b) ... (a)
% O_t = V * h_t + c ..................... (b)
global RNN
RNN.m = 100;     % # hidden states
RNN.k = INFO.K;  % # input == # output (one-hot code length)
RNN.b = zeros(RNN.m, 1);    % (a) bias vector
RNN.c = zeros(RNN.k, 1);    % (b) bias vector

RNN.sig = 0.01;  % initial param var
RNN.U = randn(RNN.m, RNN.k) * RNN.sig ;  % (a) input weight
RNN.W = randn(RNN.m, RNN.m) * RNN.sig ;  % (a) hidden weight
RNN.V = randn(RNN.k, RNN.m) * RNN.sig ;  % (b) hidden weight

RNN.seq_len = 25;

%% simple test: randomly generate text
test_length = 30;
coder = onehotCoder(RNN.k);
ind_x0 = 11;  % full stop ind
X = coder.encode(ind_x0);
rnn_test = RecurrentTest(RNN);
rnn1 = rnn_test.randomGenerate(X, test_length);
text_cell = rnn1.output();
text_mat = cell2mat(text_cell');
text_ind = int32(coder.decode(text_mat));
text_str = "";
for i = 1: numel(text_ind)
    text_str = text_str + INFO.idx2char(text_ind(i));
end
assert(strlength(text_str)==test_length);

%% load data: cell_X, cell_Y
% processBatch();
load('info/cell_data.mat');

%% simple test: forward pass
X_chars = vec_X(:, 1:RNN.seq_len);
Y_chars = vec_Y(:, 1:RNN.seq_len);
rnn = Recurrent(RNN);
rnn = rnn.forward(X_chars);

%% optimization parameters
global ADA
ADA.lr = 0.1;
ADA.n_epoch = 20;



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
% Summary: search range: 1e-4 ~ 1e-2, number of cycles: 2
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
xlabel('epoch'); ylabel('validation loss');
title('loss, without BN');

figure;
for i = 1: n_sig
    plot(loss_valid_bn{i});
    hold on;
end
legend('sig=1e-1', 'sig=1e-3', 'sig=1e-4');
xlabel('epoch'); ylabel('validation loss');
title('loss, with BN');

figure;
semilogx(sigs, test_acc);
hold on;
semilogx(sigs, test_acc_bn);
legend('without BN', 'with BN');
title('test accuracy');

