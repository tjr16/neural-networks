function J = ComputeCost(X, Y, W, b, lambda)
% A function that computes the cost function with
% L2 regularizationfor a set of images.
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

    % compute probability
    P = EvaluateClassifier(X, W, b);    % size: 10 X 10000
    
    % compute loss
    sizeY = size(Y);
    if sizeY(1) == 1
        % to linear indices
        lin_idx = sub2ind(size(P), trainy, 1:10000);
        py = P(lin_idx);
    elseif sizeY(1) == 10
        py = sum(Y .* P, 1);     
    else
        disp('Size error!');
        return
    end
    
    loss = mean(-log(py));
    penalty = lambda * sum(W .^ 2, 'all');
    J = loss + penalty;

end
