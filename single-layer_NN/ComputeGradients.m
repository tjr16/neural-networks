function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, svm)
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
%   svm: whether using SVM loss, bool
% Return:
%   grad_W: k X d (10 X 3072), partial J / partial W
%   grad_b: k X 1, partial J / partial W

    k = size(P, 1);     % class num
    nb = size(X, 2);    % batch size
    
    if svm
        Py = repmat(sum(P .* Y, 1), [k, 1]);
        G = P - Py + 1 > 0;
        G = G .* ~Y;    % remove s_y (computed as s_j)
        G_sy = - repmat(sum(G, 1), [k, 1]) .* Y;    % compute s_y
        G = G + G_sy;   % add s_y
    else
        G = P - Y;  % error (partial L / partial z)        
    end
    
    grad_W = G * X'/ nb + 2 * lambda * W;
    grad_b = G * ones(nb, 1) / nb;
end
