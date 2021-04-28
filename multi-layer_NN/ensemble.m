function acc = ensemble(nns, data)
% A function that ensembles several networks and gives
% a output value.
% ----------
% Arguments:
% 	nns: a cell of trained Doublelayers, 1 X n cell
%   data: 1 X 3 cell
% Return:
%   acc: accuracy of each network and finally the ensembled
%        result. 1 X (n+1) cell
    
    n = numel(nns);           % network amount
    nb = size(data{1}, 2);    % data amount
    P = cell(1, n);           % network output
    pred = cell(1, n);        % class prediction
    pred_all = zeros(n, nb);  % concat all pred, n X k*nb
    acc = cell(1, n+1);

    for i = 1: n
        nn = nns{i}.forward(data{1});
        P{i} = nn.output();    % k X nb
        [~, pred{i}] = max(P{i}, [], 1);   % 1 X nb
        pred_all(i, :) = pred{i};
        acc{i} = computeAccuracy(P{i}, data{3});
    end

    pred_final = mode(pred_all, 1);       % ensembled pred, 1 X k*nb
    assert(all(size(pred_final)==size(data{3})));
    acc{end} = mean(pred_final == data{3});

end