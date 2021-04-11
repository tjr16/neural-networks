function [net, metrics] = miniBatchGD(train_data, valid_data, net)
% A function that performs mini-batch gradient descent.
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
%   net: trained network
%   metrics: [loss_train; loss_valid; cost_train; cost_valid;
%       acc_train; acc_valid]   (6 X n_epoch)

    % read data
    X = train_data{1}; Y = train_data{2}; y = train_data{3};
    X_val = valid_data{1}; Y_val = valid_data{2}; y_val = valid_data{3};
    
    % parameters for optimization algorithm
    global GD2
    n_batch = GD2.n_batch;
    eta = GD2.lr;
    n_epoch = GD2.n_epoch;
    lambda = GD2.lambda;
    ns = GD2.ns;    % stepsize
    n = size(X, 2);
    n_steps = n_epoch * n/n_batch;	% #update steps
     
    % cyclic learning rate
    if GD2.cyclic
        % get learning rate array
        arr1 = linspace(0, 1, ns+1);
        arr2 = linspace(1, 0, ns+1);
        arr = [arr1(1:end-1), arr2(1:end-1)];
        etas = repmat(arr, [1, ceil(n_steps/2/ns)]);
        etas = etas(1:n_steps); % numel == n_steps
        etas = etas * (GD2.lr_max - GD2.lr) + GD2.lr;
    end
    
    % save curve: loss, cost, acc
    loss = zeros(1, n_epoch);
    cost = zeros(1, n_epoch);
    acc = zeros(1, n_epoch);
    loss_val = zeros(1, n_epoch);
    cost_val = zeros(1, n_epoch);
    acc_val = zeros(1, n_epoch);
    
    % epoch
    for i = 1: n_epoch
        random_idx = randperm(n);
        
        % batch
        for j = 1: n/n_batch
            % get batch data
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            idx = random_idx(j_start:j_end);
            Xbatch = X(:, idx);
            Ybatch = Y(:, idx);
            
            % run network and compute gradients
            net = net.computeGradients(Xbatch, Ybatch, lambda);
 
            % get learning rate
            if GD2.cyclic
                idx = (i-1) * n/n_batch + j;
                eta = etas(idx);
            end
            
            % update network parameters
            net.W{1} = net.W{1} - eta * net.grad_W{1};
            net.W{2} = net.W{2} - eta * net.grad_W{2};
            net.b{1} = net.b{1} - eta * net.grad_b{1};
            net.b{2} = net.b{2} - eta * net.grad_b{2};
        end
        
        % get output
        net_train = net.forward(X);
        net_val = net.forward(X_val);
        P = net_train.output();
        P_val = net_val.output();
        % compute loss and cost
        loss(i) = crossEntropy(P, Y);
        loss_val(i) = crossEntropy(P_val, Y_val);
        penalty = net.penaltyL2(lambda);
        cost(i) = loss(i) + penalty;
        cost_val(i) = loss_val(i) + penalty;
        % compute acc
        acc(i) = computeAccuracy(P, y);
        acc_val(i) = computeAccuracy(P_val, y_val);
        
        fprintf("Epoch %d, training cost: %f\n", i, cost(i));
        fprintf("\t validation cost: %f\n", cost_val(i));
    end
    
    metrics = [loss; loss_val; cost; cost_val; acc; acc_val];
    
end