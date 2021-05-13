function [J, dummy] = ComputeLoss(X, Y, RNN_try, hprev)
% NOTE: This function is only designed for numerical
% gradient computation. -- J.T.
% ----------
% Arguments:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   Y: k X n (10 X 10000) or 1 X n (1 X 10000)
%      one-hot or integer
%   W, b: network parameters
%   lambda: penalty coefficient, hyperparamter
% Return:
%   J: sum of loss, scalar
    
    global MLP
    dummy = 0;

    W = params.W;
    b = params.b;
    bn = params.use_bn;
    
    % run network and get output
    net = MultiLayer(W, b, [], bn);
    if bn
        net = net.setBNParam(params.gammas, params.betas);
    end
    net = net.train();
    net = net.forward(X);
    P = net.output();   % k X batch_size
    
    % compute loss
    sizeY = size(Y);

    if sizeY(1) == 1
        % to linear indices
        lin_idx = sub2ind(size(P), Y, 1:sizeY(2));
        py = P(lin_idx);
    elseif sizeY(1) == MLP.d(end)
        py = sum(Y .* P, 1);     
    else
        error('Size error!');
    end
    loss = mean(-log(py));

    % compute penalty
    penalty = 0;
    for i = 1: numel(W)
        penalty = penalty + lambda * sum(W{i} .^ 2, 'all');
    end

    J = loss + penalty;
end