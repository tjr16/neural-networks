function acc_final = evaluate(net, testData)
% A function that evaluates a trained network on test set.
% ----------
% Argument:
%   net: trained network
%   testData: cell, 1 X 3
%       X: image pixel data, d X n (3072 X 10000) 
%           double, [0, 1]
%       Y: k X n (10 X 10000)
%           uint8, {0, 1}
%       y: 1 X n (1 X 10000)
%           uint8, {1, 2, ..., 10}
% Return:
%   acc_final: final accuracy on test set

    nn_eval = net.eval();
    nn_final = nn_eval.forward(testData{1});
    P_final = nn_final.output();
    acc_final = computeAccuracy(P_final, testData{3});
    fprintf("Accuracy: %f\n", acc_final);
end