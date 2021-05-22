function loss = crossEntropy(P, Y)
% This function computes cross entropy loss for RNN.
% ----------
% Arguments:
%   P: output of the network
%   Y: k X n (10 X 10000) or 1 X n (1 X 10000)
%      one-hot or integer
% Return:
%   loss: mean loss, scalar

    % compute loss
    sizeY = size(Y);
    if sizeY(1) == 1
        % to linear indices
        lin_idx = sub2ind(size(P), Y, 1:sizeY(2));
        py = P(lin_idx);
    elseif sizeY(1) ~= size(P, 1)
        error('Size error!');
    else
        py = sum(Y .* P, 1);     
    end
    loss = sum(-log(py));

end