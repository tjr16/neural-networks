function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
% A function that evaluates, for a mini-batch,
% the gradients of the cost function wrt W and b.
% ----------
% Arguments:
%   X: image pixel data, d X nb (3072 X batch_size) 
%      double, [0, 1]
%   Y: k X nb (10 X batch_size), one-hot
%   P: k X nb
%   W: network parameters
%   lambda: penalty coefficient, hyperparamter
% Return:
%   grad_W: k X d (10 X 3072), partial J / partial W
%   grad_b: k X 1, partial J / partial W

    nb = size(X, 2);  % batch size
    G = P - Y;  % error (partial L / partial z)
    grad_W = G * X'/ nb + 2 * lambda * W;
    grad_b = G * ones(nb, 1) / nb;
    
end
