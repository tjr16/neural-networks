function tests = testGradient
% Test function RNN.computeGradients

    tests = functiontests(localfunctions);
end

function err = relativeError(Ga, Gn)
    eps = 1e-6;
    abs_err = abs(Ga - Gn);
    err = mean(abs_err ./ max(eps, abs(Ga) + abs(Gn)), 'all');
end

function testNoLambdaNoBN(testCase)
    lr = 1e-4;

    
    % load data 
    global RNN
    tmp = RNN;
    RNN.m = 5;
    
    load('info/cell_data.mat', 'vec_X', 'vec_Y');
    X_chars = vec_X(:, 1:RNN.seq_len);
    Y_chars = vec_Y(:, 1:RNN.seq_len);
 
    % analytical gradient
    nn = Recurrent(RNN);
    nn = nn.computeGradients(X_chars, Y_chars);
    analytical_grad = nn.grad;
     
    % numerical gradient
    grads = ComputeGradsNum(X, Y, NetParams, lam, lr);
      
    for f = fieldnames(RNN)'
        0
    end

    for i = 1:numel()
        testCase.verifyTrue(...
            relativeError(analytical_W{i}, grads.W{i}) < 1e-6);
        testCase.verifyTrue(...
            relativeError(analytical_b{i}, grads.b{i}) < 1e-6); 
    end
    
    RNN = tmp;
end


