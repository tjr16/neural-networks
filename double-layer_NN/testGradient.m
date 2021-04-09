function tests = testGradient
% Test function DoubleLayer.computeGradients

    tests = functiontests(localfunctions);
end

function err = relativeError(Ga, Gn)
    eps = 1e-6;
    abs_err = abs(Ga - Gn);
    err = mean(abs_err ./ max(eps, abs(Ga) + abs(Gn)), 'all');
end

function testNoLambda(testCase)
    lr = 1e-5;
    lam = 0;
    
    [trainX, trainY, ~] = loadBatch('data_batch_1.mat');
    X = trainX(:, 1:10);
    Y = trainY(:, 1:10);

    [W, b] = initParam();
    nn = DoubleLayer(W, b);
    nn = nn.computeGradients(X, Y, lam);
    [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lam, lr, true);
    numerical = {grad_b{1}, grad_b{2}, grad_W{1}, grad_W{2}};
    analytical = {nn.grad_b{1}, nn.grad_b{2}, ...
        nn.grad_W{1}(1:100), nn.grad_W{2}(1:100)};
    
    for i = 1:4
        testCase.verifyTrue(...
            relativeError(numerical{i}, analytical{i}) < 1e-7);
    end

end

function testLambda(testCase)
    lr = 1e-5;
    lam = 0.1;
    
    [trainX, trainY, ~] = loadBatch('data_batch_1.mat');
    X = trainX(:, 1:10);
    Y = trainY(:, 1:10);

    [W, b] = initParam();
    nn = DoubleLayer(W, b);
    nn = nn.computeGradients(X, Y, lam);
    [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lam, lr, true);
    numerical = {grad_b{1}, grad_b{2}, grad_W{1}, grad_W{2}};
    analytical = {nn.grad_b{1}, nn.grad_b{2}, ...
        nn.grad_W{1}(1:100), nn.grad_W{2}(1:100)};
    
    for i = 1:4
        testCase.verifyTrue(...
            relativeError(numerical{i}, analytical{i}) < 1e-7);
    end
end
