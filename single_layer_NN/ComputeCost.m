function J = ComputeCost(X, Y, W, b, lambda, svm)
% A function that computes the cost function with
% L2 regularization for a set of images.
% If lambda == 0, it computes the total loss.
% ----------
% Arguments:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   Y: k X n (10 X 10000) or 1 X n (1 X 10000)
%      one-hot or integer
%   W, b: network parameters
%   lambda: penalty coefficient, hyperparamter
%   svm: whether using SVM loss, bool
% Return:
%   J: sum of loss, scalar

    if nargin < 6
        svm = false;
    end
    
    % compute probability
    P = EvaluateClassifier(X, W, b, svm);    
    % size: k X batch_size
    
    % compute loss
    sizeY = size(Y);
    if ~svm
        if sizeY(1) == 1
            % to linear indices
            lin_idx = sub2ind(size(P), trainy, 1:10000);
            py = P(lin_idx);
        elseif sizeY(1) == 10
            py = sum(Y .* P, 1);     
        else
            error('Size error!');
        end
        loss = mean(-log(py));
    
    else
        if sizeY(1) == 1
            Y = onehotencode(categorical(Y-1), 1);  % one-hot
        end
        
        if sizeY(1) == 10
            sy = sum(Y .* P, 1);    % P denotes 's'
            L = P - sy + 1;     % s - sy + 1
            L(L < 0) = 0;   % max(0, s-sy+1)
            loss_arr = sum(L, 1) - 1;   % subtract class y
            loss = mean(loss_arr);
        else
            error('Size error!');
        end
        
    end
    
    penalty = lambda * sum(W .^ 2, 'all');
    J = loss + penalty;

end
