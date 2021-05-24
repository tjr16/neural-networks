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

%% network parameters
% notations
% h_t = tanh(W * h_t-1 + U * X_t + b) ... (a)
% O_t = V * h_t + c ..................... (b)
global RNN
RNN.m = 100;     % # hidden states
RNN.k = INFO.K;  % # input == # output (one-hot code length)
RNN.sig = 0.01;  % initial param var
RNN.seq_len = 25;   % training sequence length
RNN.field_names = {'W', 'V', 'U', 'b', 'c'};
RNN.syn_len = 200;  % synthesize length
initParam();

%% optimization parameters
global ADA
ADA.lr = 0.1;
ADA.n_epoch = 8;

%% load data: vec_X, vec_Y
% processBatch();  % make datasets
% load('info/cell_data.mat');  % cell_X, cell_Y
load('info/vec_data.mat');

%% encoder and dict
global DICT
DICT.coder = onehotCoder(RNN.k);
DICT.int2char = INFO.idx2char;

%% test1: randomly generate text
% test_length = 30;
% ind_x0 = 11;  % full stop ind
% X = RNN.coder.encode(ind_x0);
% rnn_test = RecurrentTest(RNN);
% rnn1 = rnn_test.randomGenerate(X, test_length);
% text_cell = rnn1.output();
% text_mat = cell2mat(text_cell');
% text_ind = int32(coder.decode(text_mat));
% text_str = "";
% for i = 1: numel(text_ind)
%     text_str = text_str + INFO.idx2char(text_ind(i));
% end
% assert(strlength(text_str)==test_length);

%% test2: forward pass
% X_chars = vec_X(:, 1:RNN.seq_len);
% Y_chars = vec_Y(:, 1:RNN.seq_len);
% rnn = Recurrent(RNN);
% rnn = rnn.forward(X_chars);
% rnn = rnn.computeGradients(X_chars, Y_chars);
% rnn = rnn.clipGradients();

%% test3: backward gradient computation
% for i = 1: 5
%     runtests('testGradient.m');
% end

%% Main work
rnn = Recurrent(RNN);
[rnn_trained, smooth_loss, min_info] = adaGrad(vec_X, vec_Y, rnn);
figure; plot(smooth_loss); xlabel('step'); ylabel('smooth loss');

% synthesize from scratch
fprintf('Best model: step %d, smooth loss: %f\n', min_info.iter, min_info.loss);
best_rnn = min_info.model;
final_text = best_rnn.synthesize(DICT.coder.encode(11), [], 1000);
disp(final_text);
