function [Wstar, bstar, metrics] = ...
    MiniBatchGD(X, Y, X_val, Y_val, GDparams, W, b, lambda, decay, svm)
% A function that evaluates, for a mini-batch,
% the gradients of the cost function wrt W and b.
% ----------
% Arguments:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   Y: k X n (10 X 10000), one-hot
%   GDparams: minibatch GD parameters
%       n_batch: the size of the mini-batches
%       eta: the learning rate
%       n_epochs: the number of runs through the whole training set
%   W, b: network parameters
%   lambda: penalty coefficient, hyperparamter
%   decay: decay of learning rate (OPTIONAL)
%   svm: whether using SVM loss, bool
% Return:
%   Wstar, bstar: network parameters after optimization
%   metrics: [loss_train; loss_valid; cost_train; cost_valid]
%           (4 X n_epochs)

    % parameters
    if nargin < 10
        svm = false;
    end
    
    if nargin < 9
        decay = 1;
    else
        fprintf("Learning rate decay: %f\n", decay);
    end

    n_batch = GDparams(1);
    eta = GDparams(2);
    n_epochs = GDparams(3);
    n = size(X, 2);
    
    % save total loss and cost
    loss = zeros(1, n_epochs);
    cost = zeros(1, n_epochs);
    loss_val = zeros(1, n_epochs);
    cost_val = zeros(1, n_epochs);
    
    % epoch
    for i = 1: n_epochs
        random_idx = randperm(n);
        % batch
        for j = 1: n/n_batch
            % get batch data
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            idx = random_idx(j_start:j_end);
            Xbatch = X(:, idx);
            Ybatch = Y(:, idx);
            % get output
            P = EvaluateClassifier(Xbatch, W, b, svm);
            % get gradient
            [grad_W, grad_b] = ...
                ComputeGradients(Xbatch, Ybatch, P, W, lambda, svm);
            % update parameters
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        % print and save loss
        cost(i) = ComputeCost(X, Y, W, b, lambda, svm);
        cost_val(i) = ComputeCost(X_val, Y_val, W, b, lambda, svm);
        loss(i) = ComputeCost(X, Y, W, b, 0, svm);
        loss_val(i) = ComputeCost(X_val, Y_val, W, b, 0, svm);
        fprintf("Epoch %d, training cost: %f\n", i, cost(i));
        fprintf("\t validation cost: %f\n", cost_val(i));

        % learning rate decay
        eta = eta * decay;
    end
    
    Wstar = W;
    bstar = b;
    metrics = [loss; loss_val; cost; cost_val];
end
