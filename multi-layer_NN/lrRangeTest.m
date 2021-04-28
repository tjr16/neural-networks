function [etas, acc, acc_val] = lrRangeTest(train_data, valid_data, net)
% A function that finds good boundaries of learning rate in
% Cyclical Learning Rate method. Similar to miniBatchGD.
% ----------
% Arguments:
%   train_data: cell 1 X 3
%       X: image pixel data, d X n (3072 X 10000) 
%           double, [0, 1]
%       Y: k X n (10 X 10000), one-hot
%       y: 1 X n
%   valid_data: cell 1 X 3
%   net: network to be trained
% Return:
%   etas: learning rate array
%   acc: acc on training set
%   acc_val: acc on validation set

    % read data
    X = train_data{1}; Y = train_data{2}; y = train_data{3};
    X_val = valid_data{1}; Y_val = valid_data{2}; y_val = valid_data{3};
    
    % parameters for optimization algorithm
    global GD2
    n_batch = GD2.n_batch;
    n_epoch = GD2.n_epoch;
    lambda = GD2.lambda;
    ns = GD2.ns;    % stepsize
    n = size(X, 2);
    n_steps = n_epoch * n/n_batch;	% #update steps
    
    assert(ns == n_steps);
     
    % get learning rate array
    etas = linspace(GD2.lr, GD2.lr_max, ns);

    % save acc
    acc = zeros(1, ns);
    acc_val = zeros(1, ns);
    
    % each epoch
    for i = 1: n_epoch
        fprintf('Epoch %d:...\n', i);
        random_idx = randperm(n);
        
        % each iteration
        for j = 1: n/n_batch
            idx = (i-1) * n/n_batch + j;    % iteration id
            
            % get batch data
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            indices = random_idx(j_start:j_end);
            Xbatch = X(:, indices);
            Ybatch = Y(:, indices);
            
            % run network and compute gradients
            net = net.computeGradients(Xbatch, Ybatch, lambda);
 
            % get learning rate
            eta = etas(idx);
        
            % update network parameters
            net.W{1} = net.W{1} - eta * net.grad_W{1};
            net.W{2} = net.W{2} - eta * net.grad_W{2};
            net.b{1} = net.b{1} - eta * net.grad_b{1};
            net.b{2} = net.b{2} - eta * net.grad_b{2};
            
            % --- evaluation begins ---
            net.eval_mode = true;

            % get output
            net_train = net.forward(X);
            net_val = net.forward(X_val);
            P = net_train.output();
            P_val = net_val.output();

            % compute acc
            acc(idx) = computeAccuracy(P, y);
            acc_val(idx) = computeAccuracy(P_val, y_val);

            % --- evaluation ends ---
            net.eval_mode = false;
        end
        
    end
        
end