function acc = computeAccuracy(P, y)
% A function that computes the accuracy of the network's predictions.
% ----------
% Arguments:
%   P: output of the network, k X n
%   y: 1 X n (1 X 10000)
%      uint8, {1, 2, ..., 10}
% Return:
%   acc: scalar, [0, 1]

    [~, argmax] = max(P, [], 1);
    acc = mean(argmax == y);
end