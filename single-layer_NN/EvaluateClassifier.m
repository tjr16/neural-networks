function P = EvaluateClassifier(X, W, b, svm)
% A function that evaluates the network function 
% on multiple images and returns the results.
% ----------
% Arguments:
%   X: image pixel data, d X n (3072 X 10000) 
%      double, [0, 1]
%   W, b: network parameters
%   svm: whether using SVM loss, bool
% Return:
%   P: probabilities, k X n (10 X 10000)
%
% P = EvaluateClassifier(trainX(:, 1:100), W, b);
    
    if nargin < 4
        svm = false;
    end
    
    s = W * X + b;
    
    if svm
        P = s;
    else
        P = softmax(s);
    end
    
end