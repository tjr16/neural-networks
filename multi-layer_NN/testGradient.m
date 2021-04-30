function tests = testGradient
% Test function DoubleLayer.computeGradients

    tests = functiontests(localfunctions);
end

function err = relativeError(Ga, Gn)
    eps = 1e-6;
    abs_err = abs(Ga - Gn);
    err = mean(abs_err ./ max(eps, abs(Ga) + abs(Gn)), 'all');
end

function testNoLambdaNoBN(testCase)
    lr = 1e-5;
    lam = 0;
    
    % load data
    [trainX, trainY, ~] = loadBatch('data_batch_1.mat');
    X = trainX(1:14, 1:10);    % use 14 features, 10 images
    Y = trainY(:, 1:10);
    
    global MLP
    tmp = MLP;
    MLP.d = [14, 13, 12, 11, 10];
    [NetParams.W, NetParams.b] = initParam();
    NetParams.use_bn = false;   % do not use gammas and betas
    
    % analytical gradient
    nn = MultiLayer(NetParams.W, NetParams.b);
    nn = nn.computeGradients(X, Y, lam);
    analytical_W = nn.grad_W; 
    analytical_b = nn.grad_b;
    
    % numerical gradient
    grads = ComputeGradsNumSlow(X, Y, NetParams, lam, lr);
      
    for i = 1:numel(analytical_W)
        testCase.verifyTrue(...
            relativeError(analytical_W{i}, grads.W{i}) < 1e-7);
        testCase.verifyTrue(...
            relativeError(analytical_b{i}, grads.b{i}) < 1e-7); 
    end
    
    MLP = tmp;
end

function testLambdaNoBN(testCase)
    lr = 1e-5;
    lam = 0.1;
    
    % load data
    [trainX, trainY, ~] = loadBatch('data_batch_1.mat');
    X = trainX(1:14, 1:10);    % use 14 features, 10 images
    Y = trainY(:, 1:10);
    
    global MLP
    tmp = MLP;
    MLP.d = [14, 13, 12, 11, 10];
    [NetParams.W, NetParams.b] = initParam();
    NetParams.use_bn = false;   % do not use gammas and betas
    
    % analytical gradient
    nn = MultiLayer(NetParams.W, NetParams.b);
    nn = nn.computeGradients(X, Y, lam);
    analytical_W = nn.grad_W; 
    analytical_b = nn.grad_b;
    
    % numerical gradient
    grads = ComputeGradsNumSlow(X, Y, NetParams, lam, lr);
      
    for i = 1:numel(analytical_W)
        testCase.verifyTrue(...
            relativeError(analytical_W{i}, grads.W{i}) < 1e-7);
        testCase.verifyTrue(...
            relativeError(analytical_b{i}, grads.b{i}) < 1e-7); 
    end
    
    MLP = tmp;
end
