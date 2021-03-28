function P = EvaluateClassifier(X, W, b)
% A function that evaluates the network function 
% on multiple images and returns the results.
% ----------
% Arguments:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   W, b: network parameters
% Return:
%   P: probabilities, k X n (10 X 10000)
%
% P = EvaluateClassifier(trainX(:, 1:100), W, b);

    s = W * X + b;
    P = softmax(s);
    
end