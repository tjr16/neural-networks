function J = ComputeLoss(X, Y, RNN_try, hprev)
% NOTE: This function is only designed for numerical
% gradient computation. -- J.T.

    % run network and get output
    net = Recurrent(RNN_try);
    net.h0 = hprev;
    net = net.forward(X);
    P = net.output();   % k X seq_len
    
    % compute loss
    sizeY = size(Y);

    if sizeY(1) == 1
        % to linear indices
        lin_idx = sub2ind(size(P), Y, 1:sizeY(2));
        py = P(lin_idx);
    elseif sizeY(1) == RNN_try.k
        py = sum(Y .* P, 1);
    else
        error('Size error!');
    end
    loss = sum(-log(py));
    % note that for RNN, this is sum instead of mean

    J = loss;
end