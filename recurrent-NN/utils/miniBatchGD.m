function [net, metrics] = miniBatchGD(train_data, valid_data, net, augment)
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
%   augment: [bool] use data augmentation
% Return:
%   net: trained network
%   metrics: [loss_train; loss_valid; cost_train; cost_valid;
%       acc_train; acc_valid]   (6 X n_epoch)

    if nargin < 4 || isempty(augment)
        augment = false;
    end
    
    net.eval_mode = false;
    
    % read data
    X = train_data{1}; Y = train_data{2}; y = train_data{3};
    X_val = valid_data{1}; Y_val = valid_data{2}; y_val = valid_data{3};
    
    % parameters for optimization algorithm
    global OPT
    n_batch = OPT.n_batch;
    eta = OPT.lr;
    n_epoch = OPT.n_epoch;
    lambda = OPT.lambda;
    ns = OPT.ns;    % stepsize
    n = size(X, 2);
    n_steps = n_epoch * n/n_batch;	% #update steps
     
    % cyclic learning rate
    if OPT.cyclic
        % get learning rate array
        arr1 = linspace(0, 1, ns+1);
        arr2 = linspace(1, 0, ns+1);
        arr = [arr1(1:end-1), arr2(1:end-1)];
        etas = repmat(arr, [1, ceil(n_steps/2/ns)]);
        etas = etas(1:n_steps); % numel == n_steps
        etas = etas * (OPT.lr_max - OPT.lr) + OPT.lr;
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
            indices = random_idx(j_start:j_end);
            Xbatch = X(:, indices);
            Ybatch = Y(:, indices);
            
            if augment
                Xbatch = randomAugmentation(Xbatch);
            end
            
            % run network and compute gradients
            net = net.computeGradients(Xbatch, Ybatch, lambda);
 
            % get learning rate
            if OPT.cyclic
                idx = (i-1) * n/n_batch + j;
                eta = etas(idx);
            end
            
            % update network parameters: W & b
            for k = 1: net.n_layers
                net.W{k} = net.W{k} - eta * net.grad_W{k};
                net.b{k} = net.b{k} - eta * net.grad_b{k};
            end
            
            % update network parameters: gamma & beta
            if net.bn_mode
                [gamma, beta] = net.getBNParam();
                [grad_g, grad_b] = net.getBNGrad();

                for k = 1: net.n_layers - 1
                    gamma{k} = gamma{k} - eta * grad_g{k};
                    beta{k} = beta{k} - eta * grad_b{k};
                end

                net = net.setBNParam(gamma, beta);
            end

        end
        
        % --- evaluation ---
        net.eval_mode = true;
        
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
        
        % --- evaluation ends ---
        net.eval_mode = false;
    end
    
    metrics = [loss; loss_val; cost; cost_val; acc; acc_val];
    
end