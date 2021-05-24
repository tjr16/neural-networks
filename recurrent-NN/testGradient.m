function tests = testGradient
% Test function RNN.computeGradients

    tests = functiontests(localfunctions);
end

function err = relativeError(Ga, Gn)
    eps = 1e-6;
    abs_err = abs(Ga - Gn);
    err = mean(abs_err ./ max(eps, abs(Ga) + abs(Gn)), 'all');
end

function testGrad(testCase)
    lr = 1e-4;
    
    % load data 
    global RNN 
    tmp = RNN;
    RNN.m = 5;
    RNN.b = zeros(RNN.m, 1);    % (a) bias vector
    RNN.c = zeros(RNN.k, 1);    % (b) bias vector

    RNN.sig = 0.01;  % initial param var
    RNN.U = randn(RNN.m, RNN.k) * RNN.sig ;  % (a) input weight
    RNN.W = randn(RNN.m, RNN.m) * RNN.sig ;  % (a) hidden weight
    RNN.V = randn(RNN.k, RNN.m) * RNN.sig ;  % (b) hidden weight

    RNN.seq_len = 25;
    
    load('info/vec_data.mat', 'vec_X', 'vec_Y');
    X_chars = vec_X(:, 1:RNN.seq_len);
    Y_chars = vec_Y(:, 1:RNN.seq_len);
 
    % analytical gradient
    nn = Recurrent(RNN);
    nn = nn.computeGradients(X_chars, Y_chars);
    field_names = {'W', 'V', 'U', 'b', 'c'};
    g_analytical = nn.grad;
     
    % numerical gradient
    g_numerical = ComputeGradsNum(X_chars, Y_chars, RNN, lr);

    for f = field_names
        err = relativeError(g_analytical.(f{1}), g_numerical.(f{1}));
        testCase.verifyTrue(err < 5e-6);
    end
  
    RNN = tmp;
end


