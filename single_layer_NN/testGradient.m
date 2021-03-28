function tests = testGradient
% Test function `ComputeGradients`.
    tests = functiontests(localfunctions);
end

function err = relativeError(Ga, Gn)
    eps = 1e-6;
    abs_err = abs(Ga - Gn);
    err = mean(abs_err ./ max(eps, abs(Ga) + abs(Gn)), 'all');
end

function testNoLambda(testCase)
    lr = 1e-6;
    [trainX, trainY, ~] = LoadBatch('data_batch_1.mat');
    X = trainX(:, 1:10);
    Y = trainY(:, 1:10);
    W = 0.01 * randn(10, 3072);
    b = 0.01 * randn(10, 1);
    P = EvaluateClassifier(X, W, b);
    [grad_W, grad_b] = ComputeGradients(X, Y, P, W, 0);
    [grad_b1, grad_W1] = ComputeGradsNum(X, Y, W, b, 0, lr);
    testCase.verifyTrue(relativeError(grad_W, grad_W1) < 1e-4);
    testCase.verifyTrue(relativeError(grad_b, grad_b1) < 1e-4);
end

function testLambda(testCase)
    lr = 1e-6;
    lambda = 0.1;
    [trainX, trainY, ~] = LoadBatch('data_batch_1.mat');
    X = trainX(:, 1:10);
    Y = trainY(:, 1:10);
    W = 0.01 * randn(10, 3072);
    b = 0.01 * randn(10, 1);
    P = EvaluateClassifier(X, W, b);
    [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda);
    [grad_b1, grad_W1] = ComputeGradsNum(X, Y, W, b, lambda, lr);
    testCase.verifyTrue(relativeError(grad_W, grad_W1) < 1e-4);
    testCase.verifyTrue(relativeError(grad_b, grad_b1) < 1e-4);
end
