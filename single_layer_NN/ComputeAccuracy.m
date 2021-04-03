function acc = ComputeAccuracy(X, y, W, b, svm)
% A function that computes the accuracy of the network's 
% predictions given by equation (4) on a set of data. 
% ----------
% Arguments:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   y: 1 X n (1 X 10000)
%      uint8, {1, 2, ..., 10}
%   W, b: network parameters
%   svm: whether using SVM loss, bool
% Return:
%   acc: scalar, [0, 1]

    if nargin < 5
        svm = false;
    end
    
    P = EvaluateClassifier(X, W, b, svm);
    [~, argmax] = max(P, [], 1);
    acc = mean(argmax == y);

end
